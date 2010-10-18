/* sampleFromMeasure.cu
 *
 * CUDA kernel to sample from a measure
 *
 * @author Marc A. Suchard
 */

#ifndef _Included_sampleFromMeasureKernel
#define _Included_sampleFromMeasureKernel

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

/* Thread-Block design:
 * 1 thread per datum
 * Block grid(SAMPLE_BLOCK,1)
 * TODO Needs serious optimization; very naive prefix scan; poor occupancy
 * TODO Currently only works for J = 1, easy to fix.
 */

__global__ void sampleFromMeasureSmall(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						INT* outComponent, /** Resultant choice */
						int iTJ /** Total number of components*/
					) {

	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + threadIdx.x;
	const int pdfIndex = datumIndex * iTJ;

	REAL sum = 0;
	REAL measure[256]; // TODO Determine at run-time

	for(int i=0; i<iTJ; i++) {
		sum += inMeasure[pdfIndex + i];
		measure[i] = sum;
	}

	REAL randomNumber = inRandomNumber[datumIndex] * sum;

	int index;
	for(index = 0; randomNumber > measure[index] && index < iTJ; index++)
		;

	outComponent[datumIndex] = index;

}

/*
 * Thread-block design:
 * threadIdx.x counts workers within datum
 * threadIdx.y counts datum within block
 * blockIdx.x counts data block
 *
 */

__global__ void sampleFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						INT* outComponent, /** Resultant choice */
																	int iN,
						int iT,
						int iJ

					) {
	const int kTJ = iT * iJ;
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;

	const int datumIndex = blockIdx.x * SAMPLE_BLOCK + thidy;
	const int pdfIndex = datumIndex * kTJ;

	__shared__ REAL measure[SAMPLE_BLOCK][SAMPLE_DENSITY_BLOCK]; 
	__shared__ REAL sum[SAMPLE_BLOCK];

#if defined(LOGPDF)
	__shared__ REAL maxpdf[SAMPLE_BLOCK];
#endif
	
	if (thidx == 0) {
#if defined(LOGPDF)
		maxpdf[thidy] = -1000.0;
#endif
		sum[thidy] = 0;
	}

#if defined(LOGPDF)
	//first scan: get the max values
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();

		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
				REAL dcurrent = measure[thidy][i];
				if (dcurrent > maxpdf[thidy]) {
					maxpdf[thidy] = dcurrent;
				}
			}
		}
		__syncthreads();
	}
#endif
	
	//second scan: get scaled cummulative pdfs
	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {
		measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();

		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
				#if defined(LOGPDF)
					sum[thidy] += exp(measure[thidy][i]-maxpdf[thidy]);		//rescale and exp()
				#else
					sum[thidy] += measure[thidy][i];
				#endif
				measure[thidy][i] = sum[thidy];
			}
		}
		if (chunk + thidx < kTJ) /*** ADDED */
			inMeasure[pdfIndex + chunk + thidx] = measure[thidy][thidx];
		__syncthreads();
	}

	REAL randomNumber = inRandomNumber[datumIndex] * sum[thidy];

	int index = 0;

	for(int chunk = 0; chunk < kTJ; chunk += SAMPLE_DENSITY_BLOCK) {

//		if (chunk + thidx < kTJ) /*** ADDED -- Not necessary */
			measure[thidy][thidx] = inMeasure[pdfIndex + chunk + thidx];
		__syncthreads();

		if (thidx == 0) {
			for(int i=0; i<SAMPLE_DENSITY_BLOCK; i++) {
				if (randomNumber > measure[thidy][i])
					index = i + chunk + 1;
					if (index ==kTJ) {index = kTJ-1;}
			}
		}
	}

	if (thidx == 0 && datumIndex < iN) /*** ADDED */
		outComponent[datumIndex] = index;

}

cudaError_t gpuSampleFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						INT* outComponent, /** Resultant choice */
						int iN,
						int iT,
						int iJ
				) {
	
	dim3 gridSample(iN/SAMPLE_BLOCK,1);
	if (iN % SAMPLE_BLOCK != 0)
		gridSample.x +=1 ;
	dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
	sampleFromMeasureMedium<<<gridSample, blockSample>>>(inMeasure,inRandomNumber,
			outComponent,iN,iT,iJ);
	//cudaThreadSynchronize(); 
	return cudaSuccess;
}

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

__global__ void sampleFromMeasureBig(
		REAL* inMeasure, /** Precomputed measure */
		REAL* inRandomNumber, /** Precomputed random number */
		INT* outComponent, /** Resultant choice */
		int iTJ /** Total number of components*/
	) {

	const int datumIndex = blockIdx.x;
	const int pdfIndex = datumIndex * iTJ;
	const int n = iTJ;

    // Dynamically allocated shared memory for scan kernels
//    extern  __shared__  float temp[];
	__shared__ REAL temp[32];

//	__shared__ REAL randomNumber;
//	__shared__ REAL sum;

    int thid = threadIdx.x;

    int ai = thid;
    int bi = thid + (n/2);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    TEMP(ai + bankOffsetA) = inMeasure[pdfIndex + ai];
    TEMP(bi + bankOffsetB) = inMeasure[pdfIndex + bi];

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n/2; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            TEMP(bi) += TEMP(ai);
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {

    	int index = n - 1;
//    	sum = TEMP(index);
//    	randomNumber = inRandomNumber[datumIndex] * sum;
        index += CONFLICT_FREE_OFFSET(index);
        TEMP(index) = 0;
    }

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset /= 2;

        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = TEMP(ai);
            TEMP(ai) = TEMP(bi);
            TEMP(bi) += t;
        }
    }

    __syncthreads();

    // write results to global memory
    inMeasure[pdfIndex + ai] = TEMP(ai + bankOffsetA);
    inMeasure[pdfIndex + bi] = TEMP(bi + bankOffsetB);
}

//int CDPBase::sample(double* w, int n, MTRand& mt) {
//  int i;
//  double dsum = 0;
//  double *myw = new double[n];
//  for (i = 0; i < n;i++) {
//    dsum+=w[i];
//  }
//  double ldsum = log(dsum);
//  myw[0] = exp(log(w[0]) - ldsum);
//  for (i = 1; i < n;i++) {
//    myw[i] = exp(log(w[i])-ldsum) + myw[i-1];
//  }
//  double d = mt();
//  int k;
//  for(k=0;d>myw[k];k++)
//    ;
//  //there are rare cases k >=n due to round up errors
//  if (k >n-1) { k = n-1; }
//  delete [] myw;
//  return k;				//zero based index
//}


#ifdef __cplusplus
}
#endif
#endif // _Included_sampleFromMeasureKernel
//

