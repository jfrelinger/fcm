/*
 * CDPBaseCUDA_kernel.h
 *
 *  Created on: Jun 30, 2009
 *      Author: msuchard
 */

#ifndef CDPBASECUDA_KERNEL_H_
#define CDPBASECUDA_KERNEL_H_

#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"
extern "C" __global__ void mvSparseAggregateCrossProduct(
				float *x, 
				const int *rowIndices, 
				const int *indices,
				const float *y, 
				const unsigned int numRows, 
				const unsigned int numColsInput,
				const unsigned int crossIndex,
				const unsigned int numColsOutput,
				const unsigned int extraOffset,
				const unsigned int packeddim
				);
extern "C" __global__ void mvSparseAggregate2(
				float *x, 
				const int *rowIndices, 
				const int *indices,
				const float *y, 
				const unsigned int numRows, 
				const unsigned int numCols,
				const unsigned int packeddim
				);
extern "C" cudaError_t gpuMvUpdateMeanCov(
					REAL* d_iData,
					INT* iLabels,
					INT* h_rowIndices,
					INT* h_indices,
					INT* h_clusterCount,
					INT* d_rowIndices,
					INT* d_indices,
					REAL* oMean, 
					REAL* oCov, 
					int iD, 
					int iN,
					int iT,
					int iPackedD
					);

extern "C" __global__ void mvNormalPDF(
					REAL* iData, /** Data-vector; padded */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ,
					int isLogScaled
				);
extern "C" cudaError_t gpuMvNormalPDF(
					REAL* iData, /** Data-vector; padded */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ
				);

extern "C" __global__ void sampleFromMeasureSmall(
					REAL* iMeasure, /** Precomputed measure */
					REAL* iRandomNumber, /** Random number */
					INT* outComponent, /** Resultant choice */
					int iTJ /** Total number of components*/
				);

extern "C" __global__ void sampleFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						INT* outComponent, /** Resultant choice */						
						int iN,
						int iT,
						int iJ
				);
extern "C" cudaError_t gpuSampleFromMeasureMedium(
						REAL* inMeasure, /** Precomputed measure */
						REAL* inRandomNumber, /** Precomputed random number */
						INT* outComponent, /** Resultant choice */						
						int iN,
						int iT,
						int iJ
				);

extern "C" __global__ void sampleFromMeasureBig(
					REAL* iMeasure, /** Precomputed measure */
					REAL* iRandomNumber, /** Random number */
					INT* outComponent, /** Resultant choice */
					int iTJ /** Total number of components*/
				);

extern "C" __global__ void mvNormalPDFAndSample(
						REAL* inData, /** Data-vector; padded */
						REAL* inDensityInfo, /** Density info; already padded */
						REAL* inRandomNumber,
						REAL* outPDF, /** Resultant PDF */
						INT* outComponent,
						int iD, /** Not currently necessary, as DIM is hardcoded */
						int iN,
						int iT
					);

extern "C" __global__ void argMaxMeasure(
										 REAL* inMeasure,
										 INT* argMax,
										 int iN,
										 int iT
										 );

extern "C" cudaError_t gpuArgMaxMeasure( 
										REAL* inMeasure,
										INT* argMax,
										int iN,
										int iT
										);


#endif /* CDPBASECUDA_KERNEL_H_ */
