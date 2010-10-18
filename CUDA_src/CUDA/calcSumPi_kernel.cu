/* calcSumPi_kernel.cu
 *
 * Calculates the sum of a given matrix (n by T) over n.
 *
 * @author Andrew Cron
 */

#ifndef _IncludeCalcPiKernel
#define _IncludeCalcPiKernel

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void calcSumPi(
									 REAL* inMeasure,
									 int iN,
									 int iT,
									 REAL* partialSum
									 ){

	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;
	
	const int rowsToSum = SAMPLE_BLOCK * 100;
	const int rowIndex = blockIdx.x * rowsToSum + thidy;
	
	__shared__ REAL sum[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK];
	
	/* Calculated the sum of Pi (n by T) over n */
	
	int myRow;
	for(int chunk = 0; chunk < iT; chunk+=SAMPLE_DENSITY_BLOCK){
	
		sum[thidy][thidx] = 0.0;
	
		for(int currentRow = 0; currentRow < rowsToSum; currentRow+=SAMPLE_BLOCK){
			myRow = rowIndex + currentRow;
			if(myRow < iN){
				sum[thidy][thidx] += inMeasure[ myRow * iT + thidx + chunk];
			} 
		}
		__syncthreads();
		
		if(thidy==0){
			for(int i=1; i < SAMPLE_BLOCK; i++){
				sum[0][thidx] += sum[i][thidx];
			}
			if(thidx + chunk < iT){
				partialSum[blockIdx.x * iT + thidx + chunk] = sum[0][thidx];}
		}
		__syncthreads();
		
	}
									 
}

extern "C" __global__ void calcPartialSum(
										  REAL* partialSum,
										  int numBlocks,
										  int iT,
										  REAL* finalSum
										  ){
	
	const int thidx = threadIdx.x;
	
	__shared__ REAL sum[SAMPLE_DENSITY_BLOCK * 2];
	
	for(int chunk = 0; chunk < iT; chunk+=SAMPLE_DENSITY_BLOCK*2){
		
		sum[thidx] = 0;
		
		for(int i=0; i<numBlocks; i++){
			sum[thidx] += partialSum[ i*iT + thidx + chunk ]; 

}
			
		if((thidx + chunk) < iT){
			finalSum[thidx + chunk] = sum[thidx];}
				
	}
										  
										  
}


cudaError_t gpuCalcSumPi(
						REAL* inMeasure,
						int iN,
						int iT,
						REAL* partialSum,
						REAL* finalSum
						){
									
	dim3 gridSample(iN/(SAMPLE_BLOCK*100),1);
	if (iN % (SAMPLE_BLOCK * 100) != 0)
		gridSample.x +=1 ;
	dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
	partialSum = allocateGPURealMemory(gridSample.x*iT);	
	calcSumPi<<<gridSample,blockSample>>>(inMeasure,iN,iT,partialSum);
	
	dim3 blockPartialSum(SAMPLE_DENSITY_BLOCK * 2);
	cudaThreadSynchronize();
	calcPartialSum<<<1,blockPartialSum>>>(partialSum,gridSample.x,iT,finalSum);
	
	cudaThreadSynchronize();
	cudaFree(partialSum);
	return cudaSuccess;
									
									
									
}

#ifdef __cplusplus
}
#endif
#endif