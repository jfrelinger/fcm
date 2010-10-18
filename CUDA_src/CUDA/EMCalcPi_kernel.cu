/* EMCalcPi_kernel.cu
 *
 * Normalizes each row of an N by T matrix by each row sum.
 *
 * @author Andrew Cron
 */

#ifndef _Included_EMCalcPiKernel
#define _Included_EMCalcPiKernel

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void calcPi(
						REAL* inMeasure, /** Precomputed measure */
						REAL* sumMeasure,
						int iN,
						int iT
						) {
						
       const int thidx = threadIdx.x;
       const int thidy = threadIdx.y;

       const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
       const int pdfIndex = datumIndex * iT;

       __shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK];
       __shared__ REAL sum[SAMPLE_BLOCK];
	   
	   /* Calculates the Sum of the Densities */

       if (thidx == 0){
               sum[thidy] = 0;}
			   
		int chunk;
		int piIndex;
		int rowEnd;
		
		float bigNum = 10.0e30;
		float smallNum = 10.0e-15;
		
		__syncthreads();
		
		/* Get's Mixture PDF */
		chunk = 0;
       while(chunk < iT) {
			measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
			chunk += SAMPLE_DENSITY_BLOCK;
			__syncthreads();

			if (thidx == 0) {
					if(chunk > iT){
						rowEnd = SAMPLE_DENSITY_BLOCK + iT - chunk;
					} else {
						rowEnd = SAMPLE_DENSITY_BLOCK;
					}
					for(int i=0; i<rowEnd; i++) {
							sum[thidy] += measure[thidy][i];
					}
			}
			__syncthreads();
       }
	   
	   /* Get's "Pi" eg. the ratio of the measures */
	   /* Special caution is taken if both of the measures are very small. */
	   
	   for(chunk=0; chunk < iT; chunk += SAMPLE_DENSITY_BLOCK){
			piIndex = pdfIndex + chunk + thidx;
			
			if(datumIndex < iN && (chunk+thidx<iT)){
				if(sum[thidy] >= smallNum){
					inMeasure[piIndex] = inMeasure[piIndex] / sum[thidy];
				} else if(sum[thidy] < smallNum){
					if(sum[thidy] == 0.0){
						sum[thidy] = 10e-20;
						inMeasure[piIndex] = 0.0;
					} else {
						inMeasure[piIndex] = ( bigNum * inMeasure[piIndex] ) / (bigNum * sum[thidy] );

					}
				}
	   
			}
	   }
	   
	   
       if (thidx == 0 && datumIndex < iN){
               sumMeasure[datumIndex] = sum[thidy];}
			   
			   
}

cudaError_t gpuCalcPi(
					REAL* inMeasure, /** Precomputed measure */
					REAL* outMeasure,
					int iN,
					int iT
					) {

       dim3 gridSample(iN/SAMPLE_BLOCK,1);
       if (iN % SAMPLE_BLOCK != 0)
               gridSample.x +=1 ;
       dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
       calcPi<<<gridSample, blockSample>>>(inMeasure,outMeasure,iN,iT);
       cudaThreadSynchronize();
       return cudaSuccess;
	   
}

#ifdef __cplusplus
}
#endif
#endif

