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


extern "C" __global__ void EMmvNormalPDF(
					REAL* iData, /** Data-vector; padded */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ,
					int Ntotal
				);

extern "C" cudaError_t EMgpuMvNormalPDF(
					REAL* iData, /** Data-vector; Transposed */
					REAL* iDensityInfo, /** Density info; already padded */
					REAL* oMeasure, /** Resultant measure */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ,
					int Ntotal
				);
extern "C" __global__ void calcPi(
								  REAL* inMeasure, /** Precomputed measure */
								  REAL* sumMeasure,
								  int iN,
								  int iT
								  );

extern "C" cudaError_t gpuCalcPi(
								 REAL* inMeasure, /** Precomputed measure */
								 REAL* outMeasure,
								 int iN,
								 int iT
								 );

extern "C" __global__ void calcSumPi(
									 REAL* inMeasure,
									 int iN,
									 int iT,
									 REAL* partialSum,
									 REAL* finalSum
									 );

extern "C" __global__ void calcPartialSum(
										  REAL* partialSum,
										  int numBlocks,
										  int iT,
										  REAL* finalSum
										  );

extern "C" cudaError_t gpuCalcSumPi(
									REAL* inMeasure,
									int iN,
									int iT,
									REAL* partialSum,
									REAL* finalSum
									);

extern "C" __global__ void calcSigma(
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
									 );

extern "C" cudaError_t gpuCalcSigma(
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
									);


extern "C" __global__ void shiftMeasure(
										REAL* inMeasure,
										REAL* measureShift,
										int iN,
										int iT
										);

extern "C" cudaError_t gpuShiftMeasure(
									   REAL* inMeasure,
									   REAL* measureShift,
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
