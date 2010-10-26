/*
 * CDPBaseCUDA.cpp
 *
 *  Created on: Jun 29, 2009
 *      Author: msuchard, Quanli Wang
 */

#include "stdafx.h"
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#if defined(MULTI_GPU)
	#include <boost/bind.hpp>
#endif

#include "CDPBaseCUDA.h"
#include "CDPBaseCUDA_kernel.h"

#if defined(MULTI_GPU)
	using namespace boost;
#endif
	
static CUT_THREADPROC WKSamplerThread(TGPUplan *plan){
	
	bool multigpu = plan->Multithread && plan->NumDevices> 1;
	if (multigpu) {
		cutilSafeCall( cudaSetDevice(plan->device));
	}

	int N = plan->N;
	int nchunksize = N;
	if (nchunksize>NCHUNKSIZE) { nchunksize = NCHUNKSIZE;}
	if (plan->Multithread) {
		printf("allocating DX\n");
		plan->dX = allocateGPURealMemory(DATA_PADDED_DIM * (N+DATA_IN_BLOCK));			//extra memory here.
		printf("Allocating dMeanAndSigma\n");
		plan->dMeanAndSigma = allocateGPURealMemory(PACK_DIM * plan->kT * plan->kJ);
		printf("Allocating drandom number\n");
		plan->dRandomNumber = allocateGPURealMemory(N+DATA_IN_BLOCK);			
		cudaMemcpy(plan->dX,plan->h_X, N * DATA_PADDED_DIM * SIZE_REAL,cudaMemcpyHostToDevice);
		cudaMemcpy(plan->dMeanAndSigma,plan->h_MeanAndSigma ,plan->kT * PACK_DIM * SIZE_REAL, cudaMemcpyHostToDevice);	
		cudaMemcpy(plan->dRandomNumber,plan->h_Rand, N * SIZE_REAL, cudaMemcpyHostToDevice);
		printf("allocating dComponent\n");
		plan->dComponent = allocateGPUIntMemory(N); //only  for K
		printf("Allocating ddensity\n");
		plan->dDensity = allocateGPURealMemory((nchunksize+SAMPLE_BLOCK) * plan->kT * plan->kJ);
#if defined(CDP_MEANCOV)
		printf("allocating dMean\n");
		plan->dMean = allocateGPURealMemory(plan->kD * plan->kT * plan->kJ);
		printf("allocating dCov\n");
		plan->dCov = allocateGPURealMemory(plan->kD * plan->kD * plan->kT * plan->kJ);
		printf("allocating dRowInicies\n");
		plan->dRowIndices = allocateGPUIntMemory(plan->kT * plan->kJ + 1);	//move to global allocation?
		printf("allocating dIndicies\n")
		plan->dIndices = allocateGPUIntMemory(plan->N);
#endif
	}
/*
	unsigned int hTimer;
	cutilCheckError(cutCreateTimer(&hTimer));
	cutilCheckError(cutResetTimer(hTimer));
	cutilCheckError(cutStartTimer(hTimer));
*/	
	int kCumN = 0;
	for (int iChunk = 0; iChunk * nchunksize < N; iChunk++) {	//added layer to reduce GPU memory usage
		kCumN += nchunksize;
		int kNCurrent = nchunksize;
		if(kCumN > N) {
			kNCurrent = N - iChunk * nchunksize;
		}
		gpuMvNormalPDF(plan->dX+iChunk * nchunksize * DATA_PADDED_DIM,plan->dMeanAndSigma,
			plan->dDensity,DIM, kNCurrent, plan->kT * plan->kJ);
		gpuArgMaxMeasure(plan->dDensity, plan->dZ, kNCurrent,plan->kT);
#if defined(TEST_GPU)
		cudaMemcpy(plan->hDensities,plan->dDensity,kNCurrent * plan->kT * plan->kJ *SIZE_REAL, cudaMemcpyDeviceToHost);
		REAL dmax = 0;
		for (int i = 0; i < plan->kT * plan->kJ; i++) {
			if (plan->hDensities[i] > dmax) { dmax = plan->hDensities[i];}
		}
		fprintf(stderr,"%f\n", dmax);
#endif
		
		gpuSampleFromMeasureMedium(plan->dDensity,plan->dRandomNumber+iChunk * nchunksize,
			plan->dComponent+iChunk * nchunksize,kNCurrent,plan->kT,plan->kJ);
		cudaMemcpy(plan->h_Z + iChunk*nchunksize, plan->dZ, kNCurrent*SIZE_INT, cudaMemcpyDeviceToHost);
		
	}
	
	cudaMemcpy(plan->h_Component,plan->dComponent,N * SIZE_INT, cudaMemcpyDeviceToHost);
/*
	cutilCheckError(cutStopTimer(hTimer));
	printf("GPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));
*/
#if defined(CDP_MEANCOV)	
	gpuMvUpdateMeanCov(plan->dX,
		plan->h_Component,
		plan->hRowIndices,
		plan->hIndices,
		plan->hClusterCount,
		plan->dRowIndices,
		plan->dIndices,
		plan->dMean,
		plan->dCov,
		plan->kD,
		N,
		plan->kT,
		DATA_PADDED_DIM
				);
	cudaMemcpy(plan->hMean,plan->dMean,plan->kD * plan->kT * plan->kJ * SIZE_REAL, cudaMemcpyDeviceToHost);
	cudaMemcpy(plan->hCov,plan->dCov,plan->kD * plan->kD * plan->kT * plan->kJ * SIZE_REAL, cudaMemcpyDeviceToHost);
#endif

	if (plan->Multithread) {
	  cudaFree(plan->dComponent);
	  cudaFree(plan->dDensity);
	  cudaFree(plan->dX);
	  cudaFree(plan->dMeanAndSigma);
	  cudaFree(plan->dRandomNumber);
#if defined(CDP_MEANCOV)
	  cudaFree(plan->dMean);
	  cudaFree(plan->dCov);
	  cudaFree(plan->dRowIndices);
	  cudaFree(plan->dIndices);
#endif

	  /* cliburn changed 18 August 2010
	     
	     if (plan->Multithread) {
	     cudaFree(plan->dComponent);
	     cudaFree(plan->dDensity);
	     cudaFree(plan->dX);
	     #if defined(CDP_MEANCOV)
	     cudaFree(plan->dMeanAndSigma);
	     cudaFree(plan->dRandomNumber);
	     cudaFree(plan->dMean);
	     cudaFree(plan->dCov);
	     #endif
	  */

	}
	CUT_THREADEND;
}

CDPBaseCUDA::CDPBaseCUDA() {
	initializedInstance = 0;
}

CDPBaseCUDA::~CDPBaseCUDA(void) {
	finalize();
}

void CDPBaseCUDA::MakeGPUPlans(int startdevice, int numdevices) {
	//--------------------------------------------- 
	DIM = kD;
	CHD_DIM = DIM * (DIM + 1) / 2;		// Entries in the Cholesky decomposition
	MEAN_CHD_DIM = DIM * (DIM + 3) / 2	+ 2;	// Entries in mean, Cholesky decomposition, logDet and p
	LOGDET_OFFSET = DIM * (DIM + 3) / 2;
	PACK_DIM = 16;
	while (MEAN_CHD_DIM > PACK_DIM) {PACK_DIM += 16;}
	DATA_PADDED_DIM = BASE_DATAPADED_DIM;
	while (DIM > DATA_PADDED_DIM) {DATA_PADDED_DIM += BASE_DATAPADED_DIM;}
	//---------------------------------------------
	int totalSharedMemory = (DENSITIES_IN_BLOCK * PACK_DIM + DATA_IN_BLOCK * DIM) * SIZE_REAL;
	fprintf(stderr,"Shared memory required = %d\n",	totalSharedMemory );
	if (totalSharedMemory > 16000) {
		fprintf(stderr,"Program aborted\n");
		exit(1);
	}
	
	//host memory
	hX = new REAL[DATA_PADDED_DIM * kN]();
	hMeanAndSigma = new REAL[PACK_DIM * kT * kJ]();
	hComponent = new INT[kN * 2]();
	hRandomNumber = new REAL[kN]();
	hZ = new INT[kN];
#if defined(TEST_GPU)
	hDensities = new REAL[kN * kT * kJ]();
#endif

	/*
	hX = (REAL*) calloc(DATA_PADDED_DIM * kN,SIZE_REAL);
	hMeanAndSigma = (REAL*) calloc(PACK_DIM * kT * kJ, SIZE_REAL);
	hComponent = (INT*) calloc(kN * 2, SIZE_INT);
	hRandomNumber = (REAL*) calloc(kN, SIZE_REAL);
	hZ = (INT*) calloc(kN, SIZE_INT);
#if defined(TEST_GPU)
	hDensities = (REAL*) calloc(kN * kT * kJ,SIZE_REAL);
#endif
	*/

#if defined(CDP_MEANCOV)
	hMean = new REAL[kT * kJ * kD * MAX_GPU_COUNT]();
	hCov = new READ[kT * kJ * kD * kD * MAX_GPU_COUNT]();
	hIndices = new INT[kN]();
	hRowIndices = new INT[(kT*kJ+1)*MAX_GPU_COUNT]();
	hClusterCount = new INT[(kT*kJ)*MAX_GPU_COUNT]();

	/*
	hMean = (REAL*) calloc(kT * kJ * kD * MAX_GPU_COUNT,SIZE_REAL);
	hCov = (REAL*) calloc(kT * kJ * kD * kD * MAX_GPU_COUNT,SIZE_REAL);
	hIndices = (INT*) calloc(kN,SIZE_INT);
	hRowIndices = (INT*) calloc((kT*kJ+1)*MAX_GPU_COUNT,SIZE_INT);
	hClusterCount = (INT*) calloc((kT*kJ)*MAX_GPU_COUNT,SIZE_INT);
	*/
#endif

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
		plans[i].h_Rand = hRandomNumber + i * nchunksize;
		plans[i].h_Component = hComponent+ i * nchunksize;
		plans[i].h_Z = hZ+ i * nchunksize;
		#if defined(TEST_GPU)
			plans[i].hDensities = hDensities;
		#endif

#if defined(CDP_MEANCOV)
		plans[i].hMean = hMean + i * kT *kJ * kD;
		plans[i].hCov = hCov + i * kT *kJ * kD * kD;
		plans[i].hIndices = hIndices + i * nchunksize;
		plans[i].hRowIndices = hRowIndices + i * (kT * kJ +1);
		plans[i].hClusterCount  = hClusterCount + i * (kT * kJ);
#endif
		//
		plans[i].kT = kT;
		plans[i].kJ = kJ;
		plans[i].kD = kD;
		plans[i].device = i+startdevice;
		plans[i].Multithread = numdevices>1;	//number of virtual threads 
		plans[i].NumDevices = totaldevices;		//number of actual threads
	}

	nchunksize = plans[0].N;
	if (nchunksize>NCHUNKSIZE) { nchunksize = NCHUNKSIZE;}
	if (!plans[0].Multithread) { //in this case, we only do allocate device memory once
	printf("does this work?\n");
	fprintf(stderr,"*****GPUPlans...\n");
		plans[0].dMeanAndSigma = allocateGPURealMemory(PACK_DIM * kT * kJ);
	fprintf(stderr,"*****GPUPlans...MeanAndSigma\n");
		plans[0].dX = allocateGPURealMemory(DATA_PADDED_DIM * (plans[0].N+DATA_IN_BLOCK));
	fprintf(stderr,"*****GPUPlans...dX\n");
		plans[0].dRandomNumber = allocateGPURealMemory(plans[0].N+DATA_IN_BLOCK);
	fprintf(stderr,"*****GPUPlans...RandomNumber\n");
		plans[0].dComponent = allocateGPUIntMemory(plans[0].N * 2);
	fprintf(stderr,"*****GPUPlans...Component\n");
		plans[0].dDensity = allocateGPURealMemory((nchunksize+SAMPLE_BLOCK) * kT * kJ);
	fprintf(stderr,"*****GPUPlans...Density\n");
		plans[0].dZ = allocateGPUIntMemory(nchunksize);
	fprintf(stderr,"*****GPUPlans...Z\n");
#if defined(CDP_MEANCOV)
	printf("dMean\n");
		plans[0].dMean = allocateGPURealMemory(kT*kJ*kD);
		printf("dCov\n");
		plans[0].dCov = allocateGPURealMemory(kT*kJ*kD*kD);
		printf("dRowIndices\n");
		plans[0].dRowIndices = allocateGPUIntMemory(kT*kJ + 1);
		printf("dIndices\n");
		plans[0].dIndices = allocateGPUIntMemory(plans[0].N);
#endif
	} else {
		#if defined(MULTI_GPU)
			printf("MULTIGPU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
			for (i = 0; i < numdevices; i++) {
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dRandomNumber),SIZE_REAL * (plans[i].N+DATA_IN_BLOCK)));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dComponent),SIZE_INT*  plans[i].N * 2));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dX),SIZE_REAL * DATA_PADDED_DIM * (plans[i].N+DATA_IN_BLOCK))); 
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dMeanAndSigma), SIZE_REAL * PACK_DIM * kT * kJ));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dDensity),SIZE_REAL * (nchunksize+SAMPLE_BLOCK) * kT * kJ));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dZ),SIZE_INT*plans[i].N));
#if defined(CDP_MEANCOV)
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dMean),SIZE_REAL * kD * kT * kJ));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dCov),SIZE_REAL * kD * kT * kJ * kD));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dRowIndices),SIZE_INT * (kT*kJ + 1)));
				workers[i]->call(bind(cudaMalloc, (void**)((void*)&plans[i].dIndices),SIZE_INT * plans[i].N));
#endif
			}				
		#endif
	}
}

int CDPBaseCUDA::initializeInstance(
						const int iFirstDevice,
						const int iJ,
						const int iT,
						const int iN,
						const int iD,
						const int iNumberDevices
						) {

	kJ = iJ;
	kT = iT;
	kN = iN;
	kD = iD;
	
	fprintf(stderr,"Attempting to initialize GPU device(s)...\n");
	int totalNumDevices = getGPUDeviceCount();
	if (totalNumDevices == 0) {
		fprintf(stderr,"No GPU devices found!\n");
		exit(-1);
	}

	int kUseMultipleDevices = (iNumberDevices == -1) || (iNumberDevices >= 1);
	#if defined(MULTI_GPU)
		if (totalNumDevices == 1) {
			kUseMultipleDevices = 0;
		}
	#endif

	if (!kUseMultipleDevices) {
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
// 			workers = (GPUWorker**) malloc(kDeviceCount * sizeof(GPUWorker*));
		        workers = new GPUWorker[kDeviceCount];
			for(int i=0; i<kDeviceCount; i++) {
				workers[i] = new GPUWorker(iFirstDevice +i);
				printGPUInfo(iFirstDevice + i);
			}
		#endif

	}
	
	fprintf(stderr,"Attempting to allocate memory...\n");
	kStartDevice = iFirstDevice;
	cudaSetDevice(iFirstDevice);
	MakeGPUPlans(kStartDevice,kDeviceCount);
	
	#ifdef CHECK_GPU
		checkCUDAError("Post-initialization");
	#endif

	fprintf(stderr,"Memory successfully allocated.\n");
	initializedInstance = 1;

	return CUDA_SUCCESS;
}

void CDPBaseCUDA::finalize(void) {
	// fix the memory here later!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//return;
	if (plans[0].Multithread) {
#if defined(MULTI_GPU)
			for(int i=0; i<kDeviceCount; i++) {
				workers[i]->call(bind(cudaFree,plans[i].dX));
				workers[i]->call(bind(cudaFree,plans[i].dMeanAndSigma));
				workers[i]->call(bind(cudaFree,plans[i].dDensity));
				workers[i]->call(bind(cudaFree,plans[i].dComponent));
				workers[i]->call(bind(cudaFree,plans[i].dRandomNumber));
#if defined(CDP_MEANCOV)
				workers[i]->call(bind(cudaFree,plans[i].dMean));
				workers[i]->call(bind(cudaFree,plans[i].dCov));
				workers[i]->call(bind(cudaFree,plans[i].dRowIndices));
				workers[i]->call(bind(cudaFree,plans[i].dIndices));
#endif
			}
			for (int i=0; i<kDeviceCount; ++i) {
			  delete workers[i];
			}
			delete [] workers;
			workers = NULL;
		#else
			cudaThreadExit();
		#endif
	} else {
		printf("free dDesnsity\n");
		cudaFree(plans[0].dDensity);
		printf("free dX\n");
		cudaFree(plans[0].dX);
		printf("free dMeanAndSigma\n");
		cudaFree(plans[0].dMeanAndSigma);
		printf("free dRandomNumber\n");
		cudaFree(plans[0].dRandomNumber);
		printf("free dComponent\n");
		cudaFree(plans[0].dComponent);
		printf("free dZ\n");
		cudaFree(plans[0].dZ);

#if defined(CDP_MEANCOV)
		cudaFree(plans[0].dMean);
		cudaFree(plans[0].dCov);
		cudaFree(plans[0].dRowIndices);
		cudaFree(plans[0].dIndices);
#endif
		//std::cout << cudaGetErrorString(cudaThreadSynchronize()) << std::endl;
		std::cout << cudaGetErrorString(cudaThreadExit()) << std::endl;

	}

	if(initializedInstance != 0) {
	delete [] hX;
	hX = NULL;
	delete [] hMeanAndSigma;
	hMeanAndSigma = NULL;
	delete [] hComponent;
	hComponent = NULL;
	delete hRandomNumber;
	hRandomNumber = NULL;
	delete [] hZ;
	hZ = NULL;

#if defined(CDP_MEANCOV)
	delete [] hMean;
	// free(hMean);
	hMean = NULL;
	delete [] hCov;
	// free(hCov);
	hCov = NULL;
	delete [] hIndices;
	// free(hIndices);
	hIndices = NULL;
	delete [] hRowIndices;
	// free(hRowIndices);
	hRowIndices = NULL;
	delete [] hClusterCount;
	// free(hClusterCount);
	hClusterCount = NULL;
#endif

	#ifdef CHECK_GPU
		checkCUDAError("Post-finalization");
	#endif

	initializedInstance = 0;
	printf("DONE\n");
	}
}

int CDPBaseCUDA::initializeData(double** iX) {
	if(!initializedInstance) {
		fprintf(stderr,"Device is not yet initialized!\n");
		exit(-1);
	}

	fprintf(stderr,"Attempting to upload data [%d x %d]...\n",kN,kD);

	REAL* hDatumPtr = hX;

	for(int i=0; i<kN; i++) {
		for(int j=0; j<kD; j++)
			hDatumPtr[j] = (REAL) iX[i][j];
		hDatumPtr += DATA_PADDED_DIM;
	}
	
	if (!plans[0].Multithread) {
		cudaMemcpy(plans[0].dX,hX,kN * DATA_PADDED_DIM * SIZE_REAL,cudaMemcpyHostToDevice);
	} else {
		#if defined(MULTI_GPU)
			for(int i=0; i<kDeviceCount; i++) {
				// Copy potion of data onto each GPU
				workers[i]->call(bind(cudaMemcpy,plans[i].dX,plans[i].h_X,
					plans[i].N * DATA_PADDED_DIM * SIZE_REAL, cudaMemcpyHostToDevice)); 
			}
		#endif
	}

	#ifdef CHECK_GPU
		checkCUDAError("Post-data upload");
	#endif

	fprintf(stderr,"Data uploaded!\n");

	return CUDA_SUCCESS;
}

int CDPBaseCUDA::uploadRandomNumbers(MTRand& mt) {

	// TODO Generate random numbers on GPU using MT project in SDK
	for(int i=0; i<kN; i++)
		hRandomNumber[i] = (REAL) mt();
	if (!plans[0].Multithread) {
		cudaMemcpy(plans[0].dRandomNumber,hRandomNumber,kN * SIZE_REAL, cudaMemcpyHostToDevice);
	} else {
		#if defined(MULTI_GPU)
			for(int i=0; i<kDeviceCount; i++) {
				workers[i]->call(bind(cudaMemcpy,plans[i].dRandomNumber,plans[i].h_Rand,
									  plans[i].N * SIZE_REAL, cudaMemcpyHostToDevice));
			}
		#endif
	}
	return CUDA_SUCCESS;
}

int CDPBaseCUDA::uploadMeanAndSigma(
		// TODO Update to use q (J)
		RowVector& p,
		vector<RowVector>& mu,
		vector<LowerTriangularMatrix>& Sigma,
		vector<double>& logdet) {
	int i;
	/*
	double dmaxlogdet = -1.0;	
	for (i = 0; i < kT; i++) { 
		if (logdet[i] > dmaxlogdet) {
			dmaxlogdet = logdet[i];
		}
	}
	fprintf(stderr,"%e\n", 1.0 / exp(dmaxlogdet/2));
	*/

	REAL* hTmpPtr = hMeanAndSigma;
	for(i=0; i<kT; i++) {
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

int CDPBaseCUDA::sampleWK(
				RowVector& iQ,
				vector<RowVector>& iP,
				vector<RowVector>& iMu,
				vector<LowerTriangularMatrix>& iSigma,
				vector<double>& iLogDet,
				MTRand& mt,
				int *oW,
				int *oK,
				int *oZ
				) {

	uploadMeanAndSigma(iP[0], iMu,iSigma,iLogDet);
	uploadRandomNumbers(mt);
	
	if (!plans[0].Multithread) {
		WKSamplerThread(&plans[0]); //no new thread
	} else {
		#if defined(MULTI_GPU)
			int plan;
			for (plan = 0; plan < kDeviceCount; plan++) {
				workers[plan]->callAsync(bind(gpuMvNormalPDF,plans[plan].dX,
							plans[plan].dMeanAndSigma,plans[plan].dDensity,DIM, plans[plan].N,kT * kJ));
			}
			for (plan = 0; plan < kDeviceCount; plan++) {
				workers[plan]->callAsync(bind(gpuArgMaxMeasure,plans[plan].dDensity,
							plans[plan].dZ, plans[plan].N, kT));
			}
			for (plan = 0; plan < kDeviceCount; plan++) {
				workers[plan]->callAsync(bind(gpuSampleFromMeasureMedium,plans[plan].dDensity,
					plans[plan].dRandomNumber,plans[plan].dComponent,plans[plan].N,kT,kJ));
			}
			for (plan = 0; plan < kDeviceCount; plan++) {
				workers[plan]->call(bind(cudaMemcpy,plans[plan].h_Component,plans[plan].dComponent,plans[plan].N * SIZE_INT, cudaMemcpyDeviceToHost));
			}
			for (plan = 0; plan < kDeviceCount; plan++) {
				workers[plan]->call(bind(cudaMemcpy,plans[plan].h_Z,plans[plan].dZ,plans[plan].N * SIZE_INT, cudaMemcpyDeviceToHost));
			}

			#if defined(CDP_MEANCOV)	
				gpuMvUpdateMeanCov(plans[plan].dX,
					plans[plan].h_Component,
					plans[plan].hRowIndices,
					plans[plan].hIndices,
					plans[plan].hClusterCount,
					plans[plan].dRowIndices,
					plans[plan].dIndices,
					plans[plan].dMean,
					plans[plan].dCov,
					plans[plan].kD,
					plans[plan].N,
					plans[plan].kT,
					DATA_PADDED_DIM
							);
				cudaMemcpy(plans[plan].hMean,plans[plan].dMean,plans[plan].kD * plans[plan].kT * 
					plans[plan].kJ * SIZE_REAL, cudaMemcpyDeviceToHost);
				cudaMemcpy(plans[plan].hCov,plans[plan].dCov,plans[plan].kD * plans[plan].kD * 
					plans[plan].kT * plans[plan].kJ * SIZE_REAL, cudaMemcpyDeviceToHost);
			#endif

		#else
			for (int plan = 0; plan < kDeviceCount; plan++) {
				threadID[plan] = cutStartThread((CUT_THREADROUTINE)WKSamplerThread, (void *)(plans + plan));
			}
			cutWaitForThreads(threadID, kDeviceCount);
		#endif
	}
	memcpy(oK, hComponent, kN * SIZE_INT);
	memcpy(oZ, hZ, kN * SIZE_INT);
#if defined(CDP_MEANCOV)	
	for (int plan = 1; plan < kDeviceCount; plan++) {
		int i;
		for (i = 0; i < kT; i++) {
			plans[0].hClusterCount[i] += plans[plan].hClusterCount[i];
		}
		for (i = 0; i < kT * kD; i++) {
			plans[0].hMean[i] += plans[plan].hMean[i];
		}
		for (i = 0; i < kT * kD * kD; i++) {
			plans[0].hCov[i] += plans[plan].hCov[i];
		}
	}
#endif
	return CUDA_SUCCESS;
}

int CDPBaseCUDA::getGPUDeviceCount() {
	int cDevices;
	CUresult status;
	status = cuInit(0);
	if (CUDA_SUCCESS != status)
		return 0;
	status = cuDeviceGetCount(&cDevices);
	if (CUDA_SUCCESS != status)
		return 0;
	if (cDevices == 0) {
		return 0;
	}
	return cDevices;
}

void CDPBaseCUDA::printGPUInfo(int iDevice) {

	fprintf(stderr,"GPU Device Information:");

		char name[256];
		int totalGlobalMemory = 0;
		int clockSpeed = 0;

		// New CUDA functions in cutil.h do not work in JNI files
		getGPUInfo(iDevice, name, &totalGlobalMemory, &clockSpeed);
		fprintf(stderr,"\nDevice #%d: %s\n",(iDevice+1),name);
		double mem = totalGlobalMemory / 1024.0 / 1024.0;
		double clo = clockSpeed / 1000000.0;
		fprintf(stderr,"\tGlobal Memory (MB) : %3.0f\n",mem);
		fprintf(stderr,"\tClock Speed (Ghz)  : %1.2f\n",clo);
}

void CDPBaseCUDA::getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed) {
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(deviceProp));
	cudaGetDeviceProperties(&deviceProp, iDevice);
	*oMemory = deviceProp.totalGlobalMem;
	*oSpeed = deviceProp.clockRate;
	strcpy(oName, deviceProp.name);
}

