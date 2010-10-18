/* mvNormalPDFAndSample.cu
 *
 * CUDA kernel to calculate a multivariate normal density and
 * sample from the computed measure
 *
 * @author Marc A. Suchard
 */

#ifndef _Included_mvNormalPDFAndSampleKernel
#define _Included_mvNormalPDFAndSampleKernel

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

	/* Thread-Block design:
	 * 1 thread per datum*density
	 * Block grid(DATA_IN_BLOCK,DENSITIES_IN_BLOCK)
	 * DATA_IN_BLOCK = # of datum per block
	 * DENSITIES_IN_BLOCK = # of densities per block
	 *
	 * CURRENTLY ASSUME ALL DENSITIES ARE IN ONE BLOCK!!!
	 * TODO FIX!
	 *
	 */

	__global__ void mvNormalPDFAndSample(
						REAL* inData, /** Data-vector; padded */
						REAL* inDensityInfo, /** Density info; already padded */
						REAL* inRandomNumber,
						REAL* outPDF, /** Resultant PDF */
						INT* outComponent,
						int iD, /** Not currently necessary, as DIM is hardcoded */
						int iN,
						int iT
					) {

		const int thidx = threadIdx.x;
		const int thidy = threadIdx.y;

		const int dataBlockIndex = blockIdx.x * DATA_IN_BLOCK;
		const int datumIndex = dataBlockIndex + thidx;

		const int densityBlockIndex = blockIdx.y * DENSITIES_IN_BLOCK;
		const int densityIndex = densityBlockIndex + thidy;

		const int pdfIndex = datumIndex * iT + densityIndex;

		__shared__ REAL densityInfo[DENSITIES_IN_BLOCK][PACK_DIM];
		__shared__ REAL data[DATA_IN_BLOCK][DIM];

		__shared__ REAL measure[DATA_IN_BLOCK][DENSITIES_IN_BLOCK];
//		__shared__ INT componentBuffer[DATA_IN_BLOCK];

		// Read in data
		if (thidy < DIM )
			data[thidx][thidy] = inData[DATA_PADDED_DIM*datumIndex + thidy];

		// Read in density info
		for(int chunk = 0; chunk < PACK_DIM; chunk += DATA_IN_BLOCK) {
			if (thidx < PACK_DIM)
				densityInfo[thidy][chunk + thidx] = inDensityInfo[PACK_DIM*densityIndex
			                                                      + chunk + thidx];
		}
		__syncthreads();

		// Setup pointers
		REAL* tData = data[thidx];
		REAL* tDensityInfo = densityInfo[thidy];
		REAL* tMean = tDensityInfo;
		REAL* tSigma = tDensityInfo + DIM;
		REAL  tP = tDensityInfo[LOGDET_OFFSET];
		REAL  tLogDet = tDensityInfo[LOGDET_OFFSET+1];

		// Do density calculation
		REAL discrim = 0;
		REAL xx[DIM];
		for(int i=0; i<DIM; i++) {
			xx[i] = tData[i] - tMean[i];
			REAL sum = 0;
			for(int j=0; j<=i; j++) {
				sum += *tSigma++ * xx[j]; // xx[j] is always calculated since j <= i
			}
			discrim += sum * sum;
		}

	    REAL d = exp(log(tP) -0.5 * (discrim + tLogDet + (DIM*LOG_2_PI)));

	    // Save to global memory
//		if (datumIndex < iN & densityIndex < iT)
//			outPDF[pdfIndex] = d;  // Apparently has little affect on performance

		// Save to shared memory
	    REAL* tMeasure = measure[thidx];
		tMeasure[thidy] = d;

		__syncthreads();

		// Stupid reduction
		// TODO Use parallel reduction via threads y
		if( thidy == 0 ) {
			REAL sum = 0;
			for(int i=0; i<DENSITIES_IN_BLOCK; i++) {
				sum += tMeasure[i];
			}
			REAL randomNumber = inRandomNumber[datumIndex] * sum;
			int index = 0;
			// These next two lines are the real time sink
			for(index = 0; randomNumber > tMeasure[index] && index < DENSITIES_IN_BLOCK-1; index++)
				randomNumber -= tMeasure[index];
			outComponent[datumIndex] = index;
		}
	}



#ifdef __cplusplus
}
#endif
#endif // _Included_mvNormalPDFAndSampleKernel
//

