/* EMCDPBaseCUDA.cpp
 *
 * Main CUDA Class for EM CDP algorithm.
 *
 * @author Andrew Cron
 */
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math.h>

#if defined(MULTI_GPU)
#include <boost/bind.hpp>
#endif


#include "EMCDPBaseCUDA.h"
#include "EMCDPBaseCUDA_kernel.h"
#include "CDPBaseCUDA_kernel.h"

#if defined(MULTI_GPU)
using namespace boost;
#endif

#include <iostream>
#include <iomanip>
using namespace std;

#include "newmatio.h"
#include "newmatap.h"

#include "cublas.h"



static CUT_THREADPROC updatePiThread(TGPUEMplan *plan){
	
	int N = plan->N;
	int T = plan->kT;
	int nchunksize = N;
	
	// REAL* tmpSum = (REAL*) calloc(plan->kT, SIZE_REAL);
	REAL* tmpSum = new REAL[plan->kT]();
	for (int i =0; i<T; i++) {plan->h_SumPi[i] = 0.0;}
	
	if (nchunksize>NCHUNKSIZE) { nchunksize = NCHUNKSIZE;}
	
	int kCumN = 0;
	for (int iChunk=0; iChunk*nchunksize < N; iChunk++) {
		kCumN += nchunksize;
		int currentN = iChunk * nchunksize;
		int kNCurrent = nchunksize;
		if(kCumN > N){
			kNCurrent = N - currentN;
		}
		EMgpuMvNormalPDF(plan->dX+currentN,plan->dMeanAndSigma,
					   plan->dDensity,DIM, kNCurrent, T * plan->kJ, N);
		if (plan->lastIt) {
			gpuArgMaxMeasure(plan->dDensity, plan->dZ,kNCurrent,T);
			cudaMemcpy(plan->h_Z + currentN, plan->dZ, kNCurrent*SIZE_INT, cudaMemcpyDeviceToHost);
		} else {
			gpuShiftMeasure(plan->dDensity,plan->dShift,kNCurrent,T);
			//cudaMemcpy(plan->h_pi + currentN*(plan->kT), plan->dDensity,kNCurrent*plan->kT*SIZE_REAL,cudaMemcpyDeviceToHost);
			//if(iChunk==0){std::cout << *(plan->h_pi) << *(plan->h_pi +1 ) << *(plan->h_pi+2) << endl;}
			gpuCalcPi(plan->dDensity, plan->dLikelihood, kNCurrent, T);
			gpuCalcSumPi(plan->dDensity, kNCurrent, T, plan->dPartialSum, plan->dSumPi);
			
			cudaMemcpy(tmpSum, plan->dSumPi, T*SIZE_REAL, cudaMemcpyDeviceToHost);
			for (int i=0; i<T; i++) {plan->h_SumPi[i] += tmpSum[i];}
			
			//if(plan->Chunking){
				cudaMemcpy(plan->h_pi + currentN*(plan->kT), plan->dDensity,kNCurrent*plan->kT*SIZE_REAL,cudaMemcpyDeviceToHost);
			//}
			cudaMemcpy(plan->h_Likelihood + currentN, plan->dLikelihood,kNCurrent*SIZE_REAL,cudaMemcpyDeviceToHost);
			cudaMemcpy(plan->h_shift + currentN, plan->dShift,kNCurrent*SIZE_REAL,cudaMemcpyDeviceToHost);
		}
		

	}
	delete [] tmpSum;
	// free(tmpSum);
	tmpSum = NULL;
	CUT_THREADEND;
	
}

static void subsetPi(REAL* pi, REAL* piSub, int T, int N, int currentT, int kTCurrent){
	
	for (int i=0; i<N; i++) {
		for (int t=0; t<kTCurrent; t++) {
			piSub[i*kTCurrent + t] = pi[i*T + (t+currentT)];
		}
	}
	
}

static void subsetPiTrans(REAL* pi, REAL* piSub, int T, int N, int currentN, int kNCurrent){
	
	for (int i=0; i<kNCurrent; i++) {
		for (int t=0; t<T; t++) {
			piSub[t*kNCurrent + i] = pi[(i+currentN)*T + t];
		}
	}
	
}



void EMCDPBaseCUDA::updatePi(RowVector& iP,
							 vector<RowVector>& Mu,
							 vector<LowerTriangularMatrix>& Sigma,
							 vector<double>& LogDet,
							 double* SumPi,
							 double* Likelihood
							 ){
	int sumGrid;
	if (plans[0].N % SAMPLE_BLOCK != 0) {
		sumGrid = plans[0].N / SAMPLE_BLOCK + 1;
	} else {
		sumGrid = plans[0].N / SAMPLE_BLOCK;
	}
	
	uploadMeanAndSigmaEM(iP, Mu, Sigma, LogDet);
	
	updatePiThread(&plans[0]);

	for (int i=0; i<plans[0].kT; i++) {
		SumPi[i] = (double)plans[0].h_SumPi[i];
		if (SumPi[i]==0.0) {SumPi[i] = 1.0e-35;}
	}
	for (int i=0; i<plans[0].N; i++) {
		//if(i<10){std::cout << (double)plans[0].h_Likelihood[i]<< " " << plans[0].h_shift[i] << endl;}
		*Likelihood += log((double)plans[0].h_Likelihood[i])+(double)plans[0].h_shift[i];
	}
	
}

void EMCDPBaseCUDA::updateLastPi(RowVector& iP,
								 vector<RowVector>& Mu,
								 vector<LowerTriangularMatrix>& Sigma,
								 vector<double>& LogDet,
								 int* Z
								 ){
	int sumGrid;
	if (plans[0].N % SAMPLE_BLOCK != 0) {
		sumGrid = plans[0].N / SAMPLE_BLOCK + 1;
	} else {
		sumGrid = plans[0].N / SAMPLE_BLOCK;
	}
	
	uploadMeanAndSigmaEM(iP, Mu, Sigma, LogDet);
	plans[0].lastIt = true;
	updatePiThread(&plans[0]);
	
	for (int i=0; i<plans[0].N; i++) {
		Z[i] = (int)plans[0].h_Z[i];
	}
	
}


void EMCDPBaseCUDA::updateXbar(Matrix& xBar, RowVector& SumPi){
	
	int N = plans[0].N;
	int T = plans[0].kT;
	int D = kD;
	int nchunksize = N;
	int tchunksize = T;
	int t;
	
	double chunkRatio;
	
	// REAL* tmpSum = (REAL*) calloc(T*D, SIZE_REAL);
	REAL* tmpSum = new REAL[T*D]();
	for (int i=0; i<T; i++) {
		for (int j=0; j<D; j++) {
			xBar[i][j] = 0.0;
		}
	}
	
	if (nchunksize>NCHUNKSIZE) { nchunksize = NCHUNKSIZE;}
	
	chunkRatio = (double) nchunksize / (double) N;
	tchunksize = (int) floor(chunkRatio * (double) T);
	if (tchunksize==0) {tchunksize=1;}


	int kCumT = 0;
	for (int iChunk=0; iChunk*tchunksize < T; iChunk++) {
		kCumT += tchunksize;
		int currentT = iChunk * tchunksize;
		int kTCurrent = tchunksize;
		if(kCumT > T){
			kTCurrent = T - currentT;
		}
		
		/** Copies a subset of the COLUMNS to the device */
		//if(plans[0].Chunking){
			subsetPi(plans[0].h_pi, plans[0].h_piSub, T, N, currentT, kTCurrent);
			cudaMemcpy(plans[0].dDensity, plans[0].h_piSub, kTCurrent*N*SIZE_REAL, cudaMemcpyHostToDevice);
		//}
		

		cublasSgemm('n','n', kTCurrent, D, N, 1.0, plans[0].dDensity, kTCurrent, plans[0].dX, N, 0.0, plans[0].dXBar, kTCurrent);
		
		cudaMemcpy(tmpSum, plans[0].dXBar, kTCurrent*D*SIZE_REAL, cudaMemcpyDeviceToHost);
		
		
		for (int i=0; i<kTCurrent; i++) {
			t = i+currentT;
			for (int j=0; j<D; j++) {
				xBar[t][j] = ((double)tmpSum[j*kTCurrent + i]) / SumPi[t];
			}
		}
		
	}
	
	delete [] tmpSum;
	// free(tmpSum);
	tmpSum = NULL;
	
	
	
}

void EMCDPBaseCUDA::updateSigma(vector<SymmetricMatrix>& Sigma,
				 RowVector& SumPi,
				 SymmetricMatrix& Phi,
				 Matrix& xBar,
				 double gamma,
				 double nu,
				RowVector eta){
	
	int N = plans[0].N;
	int T = plans[0].kT;
	int D = kD;
	int nchunksize = N;
	int tchunksize = T;
	int i,t,p,q;
	
	double xbarp, xbarq;
	
	int SIGMA_GRID_SIZE;
	
	if (nchunksize>NCHUNKSIZE) { nchunksize = NCHUNKSIZE;}
	
	SIGMA_GRID_SIZE = nchunksize / (SIGMA_BLOCK_SIZE * SIGMA_THREAD_SUM_SIZE);
	if (nchunksize % (SIGMA_BLOCK_SIZE*SIGMA_THREAD_SUM_SIZE)!=0) { SIGMA_GRID_SIZE += 1;}
	
	REAL* dSigmaPQ = allocateGPURealMemory(SIGMA_GRID_SIZE);
	// REAL* PartialSigmaPQ = (REAL*) calloc(SIGMA_GRID_SIZE,SIZE_REAL);
	REAL* PartialSigmaPQ = new REAL[SIGMA_GRID_SIZE]();

	//std::cout << "HERE" << endl;
	int kCumN = 0;
	for (int iChunk=0; iChunk*nchunksize < N; iChunk++) {
		kCumN += nchunksize;
		int currentN = iChunk * nchunksize;
		int kNCurrent = nchunksize;
		if(kCumN > N){
			kNCurrent = N - currentN;
		}
		
		SIGMA_GRID_SIZE = kNCurrent / (SIGMA_BLOCK_SIZE * SIGMA_THREAD_SUM_SIZE);
		if(kNCurrent % (SIGMA_BLOCK_SIZE * SIGMA_THREAD_SUM_SIZE) != 0)
			SIGMA_GRID_SIZE += 1;

		//if(plans[0].Chunking){
			subsetPiTrans(plans[0].h_pi, plans[0].h_piSub, T, N, currentN, kNCurrent);
			cudaMemcpy(plans[0].dDensity, plans[0].h_piSub,kNCurrent*T*SIZE_REAL,cudaMemcpyHostToDevice);
		//}

		
		for (t=0; t<T; t++) {
			for(p=0;p<D;p++){
				for(q=p;q<D;q++){
					
					xbarp = xBar[t][p];
					xbarq = xBar[t][q];
					
					gpuCalcSigma(dSigmaPQ,
								 plans[0].dX + currentN,
								 (REAL)xbarp, (REAL)xbarq,
								 plans[0].dDensity,
								 (REAL)SumPi[t],
								 kNCurrent, N,
								 D,p,q,T,t);
					cudaMemcpy(PartialSigmaPQ,dSigmaPQ,SIGMA_GRID_SIZE*SIZE_REAL,cudaMemcpyDeviceToHost);
					if (iChunk==0) {Sigma[t](p+1,q+1)=0.0;}
					for(i=0;i<SIGMA_GRID_SIZE;i++){Sigma[t](p+1,q+1) += (double)PartialSigmaPQ[i];}
					
	//std::cout << "HERE" << endl;

				}
			}
		}
		
		
		
	}
	//std::cout << Sigma[0];
	
	for (t=0; t<T; t++) {
		for(p=0;p<D;p++){
			for(q=p;q<D;q++){
				xbarp = xBar[t][p];
				xbarq = xBar[t][q];
				Sigma[t](p+1,q+1) = ( Phi(p+1,q+1)*nu*eta[t] + Sigma[t](p+1,q+1) + (( SumPi[t] ) / (1.0 + gamma * SumPi[t] )) * xbarp * xbarq ) / ( SumPi[t] + 3.0 + nu + 2.0*(double)D);
				
			}
		}
	}
	//std::cout << "HERE" << endl;

	delete [] PartialSigmaPQ;
	// free(PartialSigmaPQ);
	PartialSigmaPQ = NULL;
	cudaFree(dSigmaPQ);
		
	
}

int EMCDPBaseCUDA::initializeInstanceEM(
									const int iFirstDevice,
									const int iJ,
									const int iT,
									const int iN,
									const int iD,
									const int iNumberDevices,
									const bool keepLastZ
									) {
	
	kJ = iJ;
	kT = iT;
	kN = iN;
	kD = iD;
	kLastZ = keepLastZ;
	
	fprintf(stderr,"Attempting to initialize GPU device(s)...\n");
	int totalNumDevices = getGPUDeviceCount();
	if (totalNumDevices == 0) {
		fprintf(stderr,"No GPU devices found!\n");
		exit(-1);
	}
	
	int kMultipleDevices = (iNumberDevices == -1) || (iNumberDevices >= 1);
#if defined(MULTI_GPU)
	if (totalNumDevices == 1) {
		kMultipleDevices = 0;
	}
#endif
	
	if (!kMultipleDevices) {
		if (totalNumDevices <= iFirstDevice) {
			fprintf(stderr,"Fewer than %d devices found!\n",(iFirstDevice+1));
			exit(-1);
		}
		printGPUInfo(iFirstDevice);
		cudaSetDevice(iFirstDevice);
		fprintf(stderr,"Device enabled!\n");
		kDeviceCount = 1;
	} else {
		if (iNumberDevices == -1) {
			kDeviceCount = totalNumDevices - iFirstDevice;
		} else {
			// in multi gpu case, kDeviceCount is the number of GPU to be used,
			// in non-multi-gpu case, kDeviceCount is the number of thread to be used
#if defined(MULTI_GPU)
			if (totalNumDevices < iNumberDevices + iFirstDevice) {
				fprintf(stderr,"Fewer than %d devices found\n",(iFirstDevice+iNumberDevices));
				exit(-1);
			}
#endif
			kDeviceCount = iNumberDevices;
		}
#if defined(MULTI_GPU)
		fprintf(stderr,"Number of device(s) used: %d \n",kDeviceCount);
// 		workers = (GPUWorker**) malloc(kDeviceCount * sizeof(GPUWorker*));
 		workers = new GPUWorker*[kDeviceCount];
		for(int i=0; i<kDeviceCount; i++) {
			workers[i] = new GPUWorker(iFirstDevice +i);
			printGPUInfo(iFirstDevice + i);
		}
#endif
	}
	
	fprintf(stderr,"Attempting to allocate memory...\n");
	kStartDevice = iFirstDevice;
	cudaSetDevice(iFirstDevice);
	MakeGPUPlansEM(kStartDevice,kDeviceCount);
	
#ifdef CHECK_GPU
	checkCUDAError("Post-initialization");
#endif
	
	fprintf(stderr,"Memory successfully allocated.\n");
	cublasInit();
	initializedInstance = 1;
	
	return CUDA_SUCCESS;
}

int EMCDPBaseCUDA::initializeDataEM(double** iX) {
	
	if(!initializedInstance) {
		fprintf(stderr,"Device is not yet initialized!\n");
		exit(-1);
	}
	
	fprintf(stderr,"Attempting to upload data [%d x %d]...\n",kN,kD);
	
	REAL* hDatumPtr = hX;
	
	/* Typecasts and Transposes Data */
	
	for (int j=0; j<kD; j++) {
		for (int i=0; i<kN; i++) {
			hDatumPtr[i] = (REAL) iX[i][j];
		}
		hDatumPtr += kN;
	}
	
	
	if (!plans[0].Multithread) {
		cudaMemcpy(plans[0].dX,hX,kN * kD * SIZE_REAL,cudaMemcpyHostToDevice);
	} else {
#if defined(MULTI_GPU)
		for(int i=0; i<kDeviceCount; i++) {
			// Copy potion of data onto each GPU
			workers[i]->call(bind(cudaMemcpy,plans[i].dX,plans[i].h_X,
								  plans[i].N * kD * SIZE_REAL, cudaMemcpyHostToDevice)); 
		}
#endif
	}
	
#ifdef CHECK_GPU
	checkCUDAError("Post-data upload");
#endif
	
	fprintf(stderr,"Data uploaded!\n");
	
	return CUDA_SUCCESS;
}

int EMCDPBaseCUDA::uploadMeanAndSigmaEM(
									RowVector& p,
									vector<RowVector>& mu,
									//vector<UpperTriangularMatrix>& Sigma,
									vector<LowerTriangularMatrix>& Sigma,
									vector<double>& logdet) {
	
	REAL* hTmpPtr = hMeanAndSigma;
	for(int i=0; i<kT; i++) {
		double* tMu = mu[i].Store();
		double* tSigma = Sigma[i].Store();
		
		for(int j=0; j<DIM; j++)
			*hTmpPtr++ = (REAL) *tMu++;
		
		for(int j=0; j<CHD_DIM; j++)
			*hTmpPtr++ = (REAL) *tSigma++;
		
		
		*hTmpPtr++ = (REAL) p[i];
		*hTmpPtr++ = (REAL) logdet[i];
		
		hTmpPtr += ((PACK_DIM) - (MEAN_CHD_DIM));
	}
	
	if (!plans[0].Multithread) {
		cudaMemcpy(plans[0].dMeanAndSigma,hMeanAndSigma,kT * PACK_DIM * SIZE_REAL, cudaMemcpyHostToDevice);	
	} else {
#if defined(MULTI_GPU)
		for(int i=0; i<kDeviceCount; i++) {
			workers[i]->call(bind(cudaMemcpy,plans[i].dMeanAndSigma,plans[i].h_MeanAndSigma,
								  kT * kJ * PACK_DIM * SIZE_REAL, cudaMemcpyHostToDevice));
		}	
#endif
	}
	return CUDA_SUCCESS;
}

void EMCDPBaseCUDA::MakeGPUPlansEM(int startdevice, int numdevices) {
	//--------------------------------------------- 
	DIM = kD;
	CHD_DIM = DIM * (DIM + 1) / 2;		// Entries in the Cholesky decomposition
	MEAN_CHD_DIM = DIM * (DIM + 3) / 2	+ 2;	// Entries in mean, Cholesky decomposition, logDet and p
	LOGDET_OFFSET = DIM * (DIM + 3) / 2;
	PACK_DIM = 16;
	while (MEAN_CHD_DIM > PACK_DIM) {PACK_DIM += 16;}
	
	//host memory
	hX = new REAL[kD * kN]();
	hMeanAndSigma = new REAL[PACK_DIM * kT * kJ]();
	hXBar = new REAL[kD * kT]();
	hZ = new int(kN);

	// hX = (REAL*) calloc(kD * kN,SIZE_REAL);
	// hMeanAndSigma = (REAL*) calloc(PACK_DIM * kT * kJ, SIZE_REAL);
	// hXBar = (REAL*) calloc(kD * kT, SIZE_REAL);
	// hZ = (INT*) calloc(kN,SIZE_INT);
	
	int totaldevices = getGPUDeviceCount();
	int nchunksize = kN / numdevices;
	
	
	while (nchunksize * numdevices < kN) {
		nchunksize++;
	}
	int i;
	int kCumN = 0;
	for (i = 0; i < numdevices; i++) {
		kCumN += nchunksize;
		int kNCurrent = nchunksize;
		if(kCumN > kN) {
			kNCurrent = kN - i * nchunksize;
		}
		plans[i].N = kNCurrent;
		plans[i].h_X = hX+i * nchunksize * DATA_PADDED_DIM;
		plans[i].h_MeanAndSigma = hMeanAndSigma;
		plans[i].lastIt = false;

		plans[i].kT = kT;
		plans[i].kJ = kJ;
		plans[i].device = i+startdevice;
		plans[i].Multithread = numdevices>1;	//number of virtual threads 
		plans[i].NumDevices = totaldevices;		//number of actual threads
	}
	
	nchunksize = plans[0].N;
	if (nchunksize>=NCHUNKSIZE) { 
		if (nchunksize>NCHUNKSIZE) {isChunking = 1;} else {isChunking = 0;}
		nchunksize = NCHUNKSIZE;
	} else {isChunking = 0;}
	
	double chunkRatio;
	int tchunksize;
	if (isChunking) {
		chunkRatio = (double) nchunksize / (double) kN;
		tchunksize = (int) floor(chunkRatio * (double) kT);
		if (tchunksize==0) {tchunksize=1;}
	} else {
		tchunksize = kT;
	}


	
	plans[0].Chunking = isChunking;
	if (!plans[0].Multithread) { //in this case, we only do allocate device memory once
		plans[0].dMeanAndSigma = allocateGPURealMemory(PACK_DIM * kT * kJ);
		plans[0].dX = allocateGPURealMemory(kD * plans[0].N);
		
		plans[0].dDensity = allocateGPURealMemory(nchunksize * kT * kJ);
		plans[0].dLikelihood = allocateGPURealMemory(nchunksize * kT * kJ);
		plans[0].dXBar = allocateGPURealMemory(kT*kD);
		plans[0].dSumPi = allocateGPURealMemory(kT);
		plans[0].dShift = allocateGPURealMemory(nchunksize);
		
		if (kLastZ) {
			plans[0].dZ = allocateGPUIntMemory(nchunksize);
		}

		//if(isChunking){
			plans[0].h_pi = (REAL*) calloc(kN * kT, SIZE_REAL);
			plans[0].h_piSub = (REAL*) calloc(kN * (tchunksize + 1), SIZE_REAL);
		//}
		//plans[0].h_pi = (REAL*) calloc(kN * kT, SIZE_REAL);
		
		plans[0].h_Likelihood = new REAL[kN]();
		plans[0].h_SumPi = new REAL[kT]();
		plans[0].h_Xbar = new REAL[kT*kD]();

		// plans[0].h_Likelihood = (REAL*) calloc(kN, SIZE_REAL);
		// plans[0].h_SumPi = (REAL*) calloc(kT, SIZE_REAL);
		// plans[0].h_Xbar = (REAL*) calloc(kT*kD, SIZE_REAL);
		
	} else {
#if defined(MULTI_GPU)
		for (i = 0; i < numdevices; i++) {
			workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dX),SIZE_REAL * DATA_PADDED_DIM * plans[i].N)); 
			workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dMeanAndSigma), SIZE_REAL * PACK_DIM * kT * kJ));
			workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dDensity),SIZE_REAL * nchunksize * kT * kJ));
		}				
#endif
	}
}


void EMCDPBaseCUDA::finalize(void) {
        return;
	printf("Here?");
	fflush(stdout);
	if (plans[0].Multithread) {
#if defined(MULTI_GPU)
		for(int i=0; i<kDeviceCount; i++) {
			workers[i]->call(bind(cudaFree,plans[i].dX));
			workers[i]->call(bind(cudaFree,plans[i].dMeanAndSigma));
			workers[i]->call(bind(cudaFree,plans[i].dDensity));
		}
		for (int i=0; i<DeviceCount; ++i) {
		  delete workers[i];
		}
		delete [] workers;
		workers = NULL;
#else
		cudaThreadExit();
#endif
	} else {
//		cudaFree(plans[0].dDensity);
//		cudaFree(plans[0].dX);
//		cudaFree(plans[0].dMeanAndSigma);
//		cudaFree(plans[0].dLikelihood);
//		cudaFree(plans[0].dXBar);
	}
	
//	free(hX); 
//	free(hMeanAndSigma);
//	free(hXBar);
//	if(isChunking){free(plans[0].h_pi);}

	
#ifdef CHECK_GPU
	checkCUDAError("Post-finalization");
#endif
	
	initializedInstance = 0;

}

EMCDPBaseCUDA::EMCDPBaseCUDA(){}

EMCDPBaseCUDA::~EMCDPBaseCUDA(void){}

