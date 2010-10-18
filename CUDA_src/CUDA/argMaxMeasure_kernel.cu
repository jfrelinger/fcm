/* shiftMeasure.cu
 *
 * CUDA kernel to shift measure to avoid underruns in BEM
 *
 * @author Andrew Cron
 */
 
#ifndef _Included_argMaxMeasureKernel
#define _Included_argMaxMeasureKernel

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Thread-Block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datam within block
 * blockIdx.x counts data block
 *
 */
 
__global__ void argMaxMeasure(
					REAL* inMeasure, /** Precomputed measure */
					INT* argMax, /** Output Space for Shift */
					int iN,
					int iT
				) {
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int kT = iT;
	const int kN = iN;
	
	const int startIndex = blockIdx.x * SAMPLE_BLOCK;
	const int datumIndex = startIndex + thidy;
	const int pdfIndex = datumIndex * kT;
	
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK];
	__shared__ REAL maxpdf[SAMPLE_BLOCK];
	__shared__ INT argmaxpdf[SAMPLE_BLOCK];
	
	int ibnd = 0;
	
	//scan measure to arg max
	for(int chunk=0; chunk < kT; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		if (thidx==0 && chunk == 0) {
			argmaxpdf[thidy] = 0;
			maxpdf[thidy] = measure[thidy][0];
		}
		__syncthreads();
		
		if (thidx == 0) {
			if( chunk + SAMPLE_DENSITY_BLOCK <= kT ){
				ibnd = SAMPLE_DENSITY_BLOCK;
			} else {
				ibnd = kT - chunk;
			}
			
			for(int i=0; i<ibnd; i++) {
				REAL dcurrent = measure[thidy][i];
				if(dcurrent > maxpdf[thidy]) {
					maxpdf[thidy] = dcurrent;
					argmaxpdf[thidy] = chunk + i;
				}
			}
		}
		__syncthreads();
	}
	
	//If works, implement in shift kernel to faster memory transactions!
	if (thidy == 0){
		for(int chunk=0; chunk < SAMPLE_BLOCK; chunk+=SAMPLE_DENSITY_BLOCK){
			ibnd = startIndex + thidx + chunk;
			if (ibnd < kN){
				argMax[ibnd] = argmaxpdf[thidx+chunk];
			}
		}
	}
	//if (thidx == 0 && datumIndex < kN){
	//	argMax[datumIndex] = argmaxpdf[thidy];
	//}
	
}

cudaError_t gpuArgMaxMeasure(
				REAL* inMeasure,
				INT* argMax,
				int iN,
				int iT
		) {
		
	dim3 gridSample(iN/SAMPLE_BLOCK,1);
	if (iN % SAMPLE_BLOCK != 0 )
		gridSample.x += 1;
	dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
	argMaxMeasure<<<gridSample, blockSample>>>(inMeasure,argMax,iN,iT);
	cudaThreadSynchronize();
	return cudaSuccess;



}

#ifdef __cplusplus
}
#endif
#endif


	