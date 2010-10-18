/* shiftMeasure.cu
 *
 * CUDA kernel to shift measure to avoid underruns in BEM
 *
 * @author Andrew Cron
 */
 
#ifndef _Included_shiftMeasureKernel
#define _Included_shiftMeasureKernel

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
 
__global__ void shiftMeasure(
					REAL* inMeasure, /** Precomputed measure */
					REAL* measureShift, /** Output Space for Shift */
					int iN,
					int iT
				) {
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	const int kT = iT;
	const int kN = iN;
	
	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
	const int pdfIndex = datumIndex * kT;
	
	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK];
	__shared__ REAL maxpdf[SAMPLE_BLOCK];
	__shared__ REAL shift[SAMPLE_BLOCK];
	
	int ibnd = 0;
	
	
	if (thidx == 0) {
		maxpdf[thidy] = inMeasure[pdfIndex];
		shift[thidy] = maxpdf[thidy];
		//maxpdf[thidy] = -1.0e+35;
		//shift[thidy] = 1.0e+35;
	}
	
	//scan measure to get min and max values
	for(int chunk=0; chunk < kT; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
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
				}
				if(dcurrent < shift[thidy]) {
					shift[thidy] = dcurrent;
				}
			}
		}
		__syncthreads();
	}
	
	//shift the measure over by the min, but make sure we don't shift too far
	for(int chunk=0; chunk <kT; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();
		
		if (thidx == 0) {
			if( chunk + SAMPLE_DENSITY_BLOCK <= kT ){
				ibnd = SAMPLE_DENSITY_BLOCK;
			} else {
				ibnd = kT - chunk;
			}
			if( shift[thidy] < maxpdf[thidy] - 40.0 ){
					shift[thidy] = maxpdf[thidy] - 40.0;}
			for(int i=0; i<ibnd; i++) {
				measure[thidy][i] = exp( measure[thidy][i] - shift[thidy] );
			}
		}
		__syncthreads();
		
		if (chunk + thidx < kT && datumIndex < kN){
			inMeasure[pdfIndex + chunk + thidx] = measure[thidy][thidx];
		}
		if (thidx == 0 && datumIndex < kN){
			measureShift[datumIndex] = shift[thidy];
		}
		__syncthreads();
	}
				
	
}

cudaError_t gpuShiftMeasure(
				REAL* inMeasure,
				REAL* measureShift,
				int iN,
				int iT
		) {
		
	dim3 gridSample(iN/SAMPLE_BLOCK,1);
	if (iN % SAMPLE_BLOCK != 0 )
		gridSample.x += 1;
	dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
	shiftMeasure<<<gridSample, blockSample>>>(inMeasure,measureShift,iN,iT);
	cudaThreadSynchronize();
	return cudaSuccess;



}

#ifdef __cplusplus
}
#endif
#endif


	