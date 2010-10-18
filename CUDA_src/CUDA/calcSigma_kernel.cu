/* calcSigma_kernel.cu
 *
 * Calculates an element of the estimated covariance matrix in BEM.
 *
 * @author Andrew Cron
 */

#ifndef _Included_calcSigma_kernel
#define _Included_calcSigma_kernel

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void calcSigma(
							REAL* Sigma,
							REAL* X,
							REAL xbarp,
							REAL xbarq,
							REAL* pi,
							REAL sumPi,
							int iN,
							int Ntotal,
							int iD,
							int p,
							int q,
							int iT,
							int t
							){
							
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	unsigned int i = bid*SIGMA_BLOCK_SIZE*2 + tid;
	const int gridSize = SIGMA_BLOCK_SIZE*2*gridDim.x;


//	__shared__ REAL xbarp;
//	__shared__ REAL xbarq;
	__shared__ REAL sum[SIGMA_BLOCK_SIZE];

	/* Calculated the p,qth element of Sigma_t */

	sum[tid] = 0.0;
	
	//if(tid == 0){xbarp = xbar[p*iT + t];}
	//if(tid == 1){xbarq = xbar[q*iT + t];}
	
	// replaced pi[i*iT + t] with pi[t*iN + i]
	
	while (i+SIGMA_BLOCK_SIZE<iN) { 
		sum[tid] += pi[t*iN + i] * (X[p*Ntotal + i] - xbarp) * (X[q*Ntotal + i] - xbarq) + pi[t*iN + i + SIGMA_BLOCK_SIZE] * (X[p*Ntotal + (i+SIGMA_BLOCK_SIZE)] - xbarp) * (X[q*Ntotal + (i+SIGMA_BLOCK_SIZE)] - xbarq);

	 	i += gridSize; }
		
	if(i<iN){
		sum[tid] += pi[t*iN + i] * (X[p*Ntotal + i] - xbarp) * (X[q*Ntotal + i] - xbarq);
	}
	
	__syncthreads();
	
	/*while (i<iN) { 
		sum[tid] += pi[i*iT + t] * (X[p*Ntotal + i] - xbarp) * (X[q*Ntotal + i] - xbarq);
		
	 	i += gridSize; }
							
	__syncthreads();*/
	
	if (SIGMA_BLOCK_SIZE >= 512) { if (tid < 256) { sum[tid] += sum[tid + 256]; } __syncthreads(); }
	if (SIGMA_BLOCK_SIZE >= 256) { if (tid < 128) { sum[tid] += sum[tid + 128]; } __syncthreads(); }
	if (SIGMA_BLOCK_SIZE >= 128) { if (tid < 64) { sum[tid] += sum[tid + 64]; } __syncthreads(); }
	if (tid < 32) {
		if (SIGMA_BLOCK_SIZE >= 64) sum[tid] += sum[tid + 32];
		if (SIGMA_BLOCK_SIZE >= 32) sum[tid] += sum[tid + 16];
		if (SIGMA_BLOCK_SIZE >= 16) sum[tid] += sum[tid + 8];
		if (SIGMA_BLOCK_SIZE >= 8) sum[tid] += sum[tid + 4];
		if (SIGMA_BLOCK_SIZE >= 4) sum[tid] += sum[tid + 2];
		if (SIGMA_BLOCK_SIZE >= 2) sum[tid] += sum[tid + 1];
	}
	if(tid==0){Sigma[bid]=sum[0];}
	
	/*if (tid == 0){
		for(i=1;i<SIGMA_BLOCK_SIZE;i++){sum[0]+=sum[i];}
		Sigma[bid] = sum[0];
	}*/
	
						
}

cudaError_t gpuCalcSigma(
							REAL* Sigma,
							REAL* X,
							REAL xbarp,
							REAL xbarq,
							REAL* pi,
							REAL sumPi,
							int iN,
							int Ntotal,
							int iD,
							int p,
							int q,
							int iT,
							int t
							){
							
	dim3 gridSigma(iN / (SIGMA_BLOCK_SIZE * SIGMA_THREAD_SUM_SIZE) );
	if(iN % (SIGMA_BLOCK_SIZE * SIGMA_THREAD_SUM_SIZE) != 0)
		gridSigma.x += 1;
	dim3 blockSigma(SIGMA_BLOCK_SIZE);
	calcSigma<<<gridSigma, blockSigma>>>(Sigma, X, xbarp,xbarq, pi, sumPi, iN, Ntotal, iD, p, q, iT, t);
	//cudaThreadSynchronize();
	return cudaSuccess;

							
}



#ifdef __cplusplus
}
#endif
#endif