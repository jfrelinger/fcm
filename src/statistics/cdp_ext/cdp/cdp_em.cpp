/* cdp_em.cpp
 *
 * Main Class for EM CDP algorithm.
 *
 * @author Andrew Cron
 */

//#define CDP_CUDA

#include <math.h>
#include <cstdio> // for debugging
#include "stdafx.h"
#include "Model.h"
#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "MersenneTwister.h"
#include "specialfunctions2.h"
#include "cdpprior.h"
#include "cdpbase.h"
#include "cdpresult.h"
#include "cdpresultem.h"
#include "cdp.h"
#include "cdp_em.h"





#if defined(CDP_TBB)
	#include "cdpt.h"
#endif



#if defined(CDP_CUDA)
	#include "CDPBaseCUDA.h"
	#include <cutil_inline.h>
	#include <cuda_runtime_api.h>
#endif

//int DIM,MEAN_CHD_DIM,PACK_DIM,CHD_DIM,LOGDET_OFFSET,DATA_PADDED_DIM,NCHUNKSIZE;



CDP_EM::CDP_EM()

{


}



CDP_EM::~CDP_EM()

{
	mX = 0;
	/*
	if (mX != 0){
	for (int i = 0; i < prior.N; i++) {
		delete [] mX[i];
		mX[i] = 0;
	}
	delete [] mX;
	mX = 0;
	};
	*/

};



bool CDP_EM::iterateEM(CDPResultEM& result,int printout)

{

	

	result.postLL = 0.0;

	

//	bool doPAlpha = true;
//	bool doMu = true;
//	bool doSigma = true;


	
	
#ifdef CDP_CUDA
	//std::cout << "Updating Pi" << endl;
	emcuda.updatePi(result.p[0], result.mu, result.L_i, result.Sigma_log_det, result.sumiPi.Store(),&result.postLL);
	//getPi_ij(result);
	//std::cout << result.sumiPi;
	//std::cout << "Updating Xbar" << endl;
	emcuda.updateXbar(result.xbar, result.sumiPi);
	//getXbar(result);
	//std::cout << result.xbar;
	
#else
	getPi_ij(result);
	//std::cout << result.sumiPi;
	getXbar(result);
#endif
	
	updatePandAlpha(result);
	updateMu(result);



#ifdef CDP_CUDA
	//std::cout << "Updating Sigma" << endl;
	emcuda.updateSigma(result.Sigma, result.sumiPi, result.Phi[0], result.xbar, prior.gamma, prior.nu, result.etaEst);
	//updateSigma(result);
#else
		updateSigma(result);
#endif
			
//	std::cout << result.Sigma[0];
	
	
	LowerTriangularMatrix temp;
	SymmetricMatrix sigmaInv;
	Matrix workMat;
	IdentityMatrix ident(result.D);
	double epsilon = 10e-2;
		
	for (int t=0; t<prior.T; t++) {
		
		
		int flag=0;
		int ITERTRY = 10;
		
		
		do{
			try {
				temp << Cholesky(result.Sigma[t]);
				flag=0;
			}
			catch (NPDException) {
				flag++;
				result.Sigma[t] << result.Sigma[t] + ident * epsilon;
				if (flag>ITERTRY) {
					std::cout << "Covariance Matrix for the "<< t<< "th component could not be made pos-def after 10 tries" << endl;
					exit(1);
				}
			}
		} while (flag > 0) ;
			
		result.Sigma_log_det[t] = msf.logdet(temp);
		result.L_i[t] = temp.i();
		
		sigmaInv << result.L_i[t].t() * result.L_i[t];
		workMat = result.Phi[0] * ( sigmaInv );

		if (emupdateeta) {
			//sigmaInv << result.L_i[t].t() * result.L_i[t];
			//workMat = result.Phi[0]*sigmaInv;
			result.etaEst[t]=(prior.aa + prior.D*(prior.nu + (double)result.D + 1.0)) / (prior.aa + prior.nu*(workMat.Trace()));
			if (result.etaEst[t] < 10e-5) {
				result.etaEst[t] = 10e-5;
			}
		}
					 
		//Some of the prior part of the posterior log-likelihood. I think this is OK.
		//result.postLL -= (prior.nu + (double)result.D + 4.0)*( result.Sigma_log_det[t] / 2.0 );
		result.postLL -= 0.5*(prior.nu + 3.0 + 2.0*(double)result.D)*result.Sigma_log_det[t];
					 
		
		if (emupdateeta) {
			//result.postLL -= (prior.D*prior.nu+2.0*(prior.D-1)+prior.aa)/2.0 * log(prior.nu*workMat.Trace()+prior.aa);
			result.postLL -= 0.5 * (prior.D*(prior.nu+prior.D+1.0)+prior.aa) * log(prior.nu*workMat.Trace()+prior.aa);
		} else {
			result.postLL -= 0.5 * prior.nu * workMat.Trace();
		}	

		result.postLL -= (1.0/(2.0 * prior.gamma)) * (( result.mu[t] - prior.m0 ) * sigmaInv * ( result.mu[t].t() - prior.m0.t() )).AsScalar();

	}	
	if (printout>0) {  
		std::cout << "log posterior = " << result.postLL << endl;
	}
	

	//For testing
	
	//ofstream pi_ijfile;
	//pi_ijfile.open("pi_ij.txt");
	//pi_ijfile << setw(15) << setprecision(10) << result.pi_ij;


	/*ofstream xbarfile;
	xbarfile.open("xbartest.txt");
	ofstream sumiPifile;
	sumiPifile.open("sumiPi.txt");

	


	xbarfile << setw(10) << setprecision(5) << result.xbar;
	sumiPifile << setw(15) << setprecision(10) << result.sumiPi;
	std::cout << "Alpha = " << result.alpha[0] << endl;
	std::cout << "Does this make any sense? " << setprecision(15) << result.postLL << endl;*/

	
	return true;

}



// Test this more carefully. Most evidence points here ....
void CDP_EM::getPi_ij(CDPResultEM& result)	

{

	int i,t;
	RowVector x_row(result.D);
	RowVector pi_row(result.T);
	RowVector shift(result.N);
	RowVector maxPdf(result.N);
	
	//LowerTriangularMatrix temp;
	double mixPdf;

	double tmp = 0.0;
	
	double bigNum = 10.0e25;
	double smallNum = 10.0e-10;

	for (i=0; i<result.N; i++) {

		for (t=0; t<result.T; t++) {

			//temp = (result.Sigma_T_i[t]).t();

			result.pi_ij[i][t] = log(result.p[0][t]) + msf.mvnormpdf(mX[i], result.mu[t].Store(), result.L_i[t], result.D, 1, result.Sigma_log_det[t]);
			//result.pi_ij[i][t] = result.p[0][t] * msf.mvnormpdf(mX[i], result.mu[t].Store(), result.L_i[t], result.D, 0, result.Sigma_log_det[t]);
			//pdf[i][t] =  msf.mvnormpdf(mX[i], result.mu[t].Store(), result.Sigma_T_i[t], result.D, 0, result.Sigma_log_det[t]);

		}		

		//if(i==0){std::cout << result.pi_ij[0][0] << endl;}
		
		shift[i] = result.pi_ij.Row(i+1).Minimum();
		maxPdf[i] = result.pi_ij.Row(i+1).Maximum();
		
		if (shift[i] < maxPdf[i] - 70.0) {
			shift[i] = maxPdf[i]-70.0;
		}
		
		for (t=0; t<result.T; t++) {
			result.pi_ij[i][t] = exp( result.pi_ij[i][t] - shift[i] );
		}
		
		
		mixPdf = result.pi_ij.Row(i+1).Sum();
		
		// Normalize Pi_ij to sum to 1

		for (t=0; t<result.T; t++) {
			//if(mixPdf >= smallNum){
			//	result.pi_ij[i][t] = result.pi_ij[i][t] / mixPdf;
			//} else if(mixPdf < smallNum){
			//	if(mixPdf == 0.0){
			//		mixPdf = 10e-20;
			//		result.pi_ij[i][t] = 0.0;
			//	} else {
			//		result.pi_ij[i][t] = ( bigNum * result.pi_ij[i][t] ) / (bigNum * mixPdf );

			//	}
			//}

			result.pi_ij[i][t] = result.pi_ij[i][t] / mixPdf;

		}
		
		//if(i<10){std::cout << mixPdf<<" " << shift[i]<<endl;}

		// Add the data-likelihood to the posterior likelihood.

		//result.postLL += log(mixPdf) + maxPdf;
		result.postLL += log(mixPdf)+shift[i];


	}
	
	//std::cout << "Like " << result.postLL << endl;

	//std::cout << setw(25) << setprecision(20) <<  pdf;

	//std::cout << "data log-likelihood " << result.postLL << endl;
		

	// Get sumiPi

	for (t=1; t<=result.T; t++) {

		result.sumiPi(t) = result.pi_ij.Column(t).Sum();
		if (result.sumiPi(t) == 0.0) {
			result.sumiPi(t) = 10e-20;
		}

	}



}




// This is just a standard matrix multiplication of pi_ij' * X but the way x is stored doesn't let me use newmat
// Looks correct.

void CDP_EM::getXbar(CDPResultEM& result){

	int t,d,n;

	double sum = 0.0;

	

	for (t=0; t<result.T; t++) {

		for (d=0; d<result.D; d++) {

			sum = 0.0;

			for (n=0; n<result.N; n++) {

				sum += result.pi_ij[n][t] * mX[n][d];

			}

			result.xbar[t][d] = sum / result.sumiPi[t];

		}

	}

		

	

}



// This updates the component weights along with V and sumiPi (needed for other computations)

// Looks correct.

void CDP_EM::updatePandAlpha(CDPResultEM& result)

{

	int t, r;

	

	double vepsilon = 0.000000000000001;

	double ww = 1.0;

	double sum = 0.0;

	

	RowVector partialSum(result.T);

	



	// Update V and Alpha (Newton's Method)

	partialSum[0] = (double)result.N;

	for (t=1; t<result.T; t++) {

		partialSum[t] = partialSum[t-1] - result.sumiPi[t-1];

	}

	

	for (r=0; r<10; r++) {

		// V

		for (t=0; t<result.T-1; t++) {

			result.pV[0][t] = result.sumiPi[t] / ( partialSum[t] + result.alpha[0] - 1.0);
			result.pV[0][t] = max(vepsilon, result.pV[0][t]);
			if (result.pV[0][t] >= 1.0) { result.pV[0][t] = 1.0 - vepsilon; }

		}
		result.pV[0][result.T-1] = 1.0;

		// Alpha

		sum = 0.0;
		for (t=0; t<result.T-1; t++) {sum += log( 1.0 - result.pV[0][t] );}
		result.alpha[0] = ( (double)result.T + prior.ee - 2.0) / (prior.ff - sum);

	}

	

	// Update component weights:	

	for (t=0; t<result.T-1; t++) {
		result.p[0][t] = ww * result.pV[0][t];
		ww = ww * (1.0 - result.pV[0][t]);

	}

	result.p[0][result.T-1] = ww;

	

	// Some of the posterior log-likelihood

	result.postLL += ( (double)result.T + prior.ee - 2.0 )*log(result.alpha[0]) - prior.ff*result.alpha[0];

	result.postLL += (result.alpha[0] - 1.0)*sum;

	//std::cout << ( (double)result.T + prior.ee - 2.0 )*log(result.alpha[0]) - prior.ff*result.alpha[0] + (result.alpha[0] - 1.0)*sum << endl;

}



void CDP_EM::updateMu(CDPResultEM& result){

	

	int t,d;

	RowVector xbarRow(result.D);

	for (t=0; t<result.T; t++) {

		for (d=0; d<result.D; d++) {

			xbarRow[d] = result.xbar[t][d];

		}

		

		result.mu[t] = ( ( prior.m0 / prior.gamma )  + xbarRow*result.sumiPi[t] ) * ( 1.0 / ( 1.0/prior.gamma + result.sumiPi[t]) ) ;

	}

	

}

extern "C" {
	
	void CPUcalcSigma(double* h_Sigma,int j, int JJ, int nn, int dd, double** h_X, double* h_xbar, double* h_pi_ij, double* h_sumipi,double* h_phi, double gamma, double nu, double eta){
		
		double sum=0.0;
		double xbarp,xbarq;
		int i,p,q;
			
			for(p=0;p<dd;p++){
				for(q=p;q<dd;q++){
					sum=0.0;
					xbarp = h_xbar[j*dd + p];
					xbarq = h_xbar[j*dd + q];	
					//printf("here");fflush(stdout);
					for(i=0;i<nn;i++){
						sum+=h_pi_ij[i*JJ + j] * (h_X[i][p] - xbarp) * (h_X[i][q] - xbarq);
					}
					//if(j==0){printf("%f\n",sum);}
					h_Sigma[p*dd+q] = ( eta*nu*h_phi[p*dd+q] + sum + (( h_sumipi[j] ) / (1.0 + gamma * h_sumipi[j] )) * xbarp*xbarq ) / ( h_sumipi[j] + nu + 3.0 + 2.0*(double)dd);
					
					h_Sigma[q*dd + p] = h_Sigma[p*dd+q];
				}
			}
		
		
		
	}
	
}



void CDP_EM::updateSigma(CDPResultEM& result)

{
	int t;
	RowVector xbarRow(result.D);
	RowVector xRow(result.D);
	RowVector diff(result.D);

	//SymmetricMatrix newSigma(result.D);
	Matrix newSigma(result.D,result.D); Matrix tmpPhi(result.D,result.D);
	SymmetricMatrix tmpMat(result.D);
	SymmetricMatrix sigmaInv(result.D);


	Matrix workMat(result.D,result.D);
	RowVector workVec(result.D);


	LowerTriangularMatrix temp;
		

	for (t=0; t<prior.T; t++) {

		
		/*newSigma = result.Phi[0];

		for (d=0; d<result.D; d++) {
			xbarRow[d] = result.xbar[t][d];
		}

		for (n=0; n<result.N; n++) {
			for (d=0; d<result.D; d++) {
				xRow[d] = mX[n][d];
			}
			diff = xRow - xbarRow;
			tmpMat <<  diff.t() * diff;
			newSigma += tmpMat * result.pi_ij[n][t] ;

		}

		diff = xbarRow - prior.m0;
		tmpMat << diff.t() * diff;
		newSigma += tmpMat * (( result.sumiPi[t] ) / (1.0 + prior.gamma * result.sumiPi[t] )) ;
		result.Sigma[t] =  newSigma / ( result.sumiPi[t] + prior.nu + (double)result.D + 2.0 );*/
		
		
		tmpPhi << result.Phi[0];
		
		CPUcalcSigma(newSigma.Store(), t , prior.T, prior.N, prior.D, mX, result.xbar.Store(), result.pi_ij.Store(), result.sumiPi.Store(),tmpPhi.Store(), prior.gamma, prior.nu, result.etaEst[t]);

		result.Sigma[t] << newSigma;
		/*temp << Cholesky(result.Sigma[t]);

		result.Sigma_log_det[t] = msf.logdet(temp);
		result.Sigma_T_i[t] = temp.t().i();

		//Some of the prior part of the posterior log-likelihood. I think this is OK.
		result.postLL -= (prior.nu + (double)result.D + 2.0)*( result.Sigma_log_det[t] / 2.0 );

	
		sigmaInv << result.Sigma_T_i[t] * result.Sigma_T_i[t].t();
		
		workMat = result.Phi[0] * ( sigmaInv );

		//result.postLL -= 0.5 * workMat.Trace();
		result.postLL -= (1.0/(2.0 * prior.gamma)) * (( result.mu[t] - prior.m0 ) * sigmaInv * ( result.mu[t].t() - prior.m0.t() )).AsScalar();*/

		
	}

	

}

void CDP_EM::EMAlgorithm(Model& model, MTRand& mt){
	
	
	
	double oldLL = 10000000000.0;
	
	double errTol;
	SymmetricMatrix sigmaInv;
	Matrix workMat;
	
	cdpfunctions.LoadData(model);
	mX = cdpfunctions.mX;

	
	prior.Init(model);
	cdpfunctions.prior = prior;
	
	CDPResultEM result(prior.J,prior.T,prior.N,prior.D);

	
#if defined(CDP_CUDA)
	
	if (model.mnGPU_Sample_Chunk_Size < 0) {
		
		NCHUNKSIZE = model.mnN;
		
	} else {
		
		NCHUNKSIZE = model.mnGPU_Sample_Chunk_Size;
		
	}
	

	emcuda.initializeInstanceEM(model.startDevice,prior.J,prior.T, prior.N, 
								  
								  prior.D,model.numberDevices); 

	emcuda.initializeDataEM(mX);
	
#endif
	cdpfunctions.SimulateFromPrior2(result,mt,1);
	// if any values are to be loaded from file, load them here
	cdpfunctions.LoadInits(model,result, mt);	
	// see if we're dealing with a special case of J==1
	cdpfunctions.CheckSpecialCases(model,result);
	
	emupdateeta = model.sampleEta;
	
	result.etaEst = RowVector(prior.T);
	
	if (emupdateeta) {
		for (int t=0; t<prior.T; t++) {
			sigmaInv << result.L_i[t].t() * result.L_i[t];
			workMat = result.Phi[0]*sigmaInv;
			result.etaEst[t]=(prior.aa - prior.nu + 2) / (prior.aa + prior.nu*(workMat.Trace()));
		}
	} else {
		for (int t=0; t<prior.T; t++) {
			result.etaEst[t]=1.0;
		}
	}

	
#if defined(CDP_CUDA)
	
	unsigned int hTimer;
	
	cutilCheckError(cutCreateTimer(&hTimer));
	
	cutilCheckError(cutResetTimer(hTimer));
	
    cutilCheckError(cutStartTimer(hTimer));
	
#else
	
	time_t tStart, tEnd;
	
	//long tStart, tEnd;
	
	//tStart = clock();
	
	time(&tStart);
	
#endif
	// main EM loop
	errTol = log(1.0 + model.mnErrorPerTol);
	
	for (int it = 0; (it < model.mnIter) & (fabs(result.postLL - oldLL) > errTol) ; it++) {
		
		oldLL = result.postLL;
		
		int printout = model.mnPrintout > 0 && it % model.mnPrintout == 0 ? 1: 0;
		if (printout>0) {  
			std::cout << "it = " << (it+1) << endl;
		}
		iterateEM(result,printout);	
	}
	
 if (model.mnPrintout >0) {
	#if defined(CDP_CUDA)
		cutilCheckError(cutStopTimer(hTimer));
		printf("GPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));
	#else
		//tEnd = clock();
		time(&tEnd);
		cout << "time lapsed:" << difftime(tEnd,tStart)  << "seconds"<< endl;
	#endif	
  }
	
	// save final parameter values
	result.SaveFinal();
	result.savePostLL("lastPostLL.txt");
}
