/*
 * mvSparseAggregate_kernel.cu
 *
 * CUDA kernel to compute aggregrate sums conditional on
 * cluster indicators
 *
 *  Created on: Oct 28, 2009
 * @author Marc A. Suchard
 */

#ifndef MVSPARSEAGGREGATE_KERNEL_CU_
#define MVSPARSEAGGREGATE_KERNEL_CU_

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE_DIM_LOG2	4
#define BLOCK_SIZE_COM_LOG2	4
#define	BLOCK_SIZE_DIM		(1<<BLOCK_SIZE_DIM_LOG2)
#define BLOCK_SIZE_COM		(1<<BLOCK_SIZE_COM_LOG2)

__global__ void	mvSparseAggregate(
								REAL* iData,
								INT*  iComponent,
								REAL* oMean,
								int   iD,
								int   iN,
								int   iTJ
							) {

	int tC = threadIdx.x;
	int tD = threadIdx.y;

	int bC = blockIdx.x;
	int bD = blockIdx.y;

	// Block specific
	int offsetComponent = bC * BLOCK_SIZE_COM;
	int offsetDimension = bD * BLOCK_SIZE_DIM;
	int offsetData      = 0;
//	int offsetMean      = 0;

	// Thread specific
	int thisComponent =	offsetComponent + tC;
	int thisDimension = offsetDimension + tD;

	__shared__ REAL X[BLOCK_SIZE_COM][BLOCK_SIZE_DIM];
	__shared__ REAL S[BLOCK_SIZE_COM][BLOCK_SIZE_DIM];

	__shared__ INT  I[BLOCK_SIZE_COM];


	// Initialize
	S[tC][tD] = 0;

	I[tC] = iComponent[thisComponent]; // TODO Are these coalesced, or do we need to index on x (not y)?

	// Loop for each datum (an easy way to start!)
	for(int i=0; i<iN; i += BLOCK_SIZE_COM) {

		// Check for end of data (N) and dimension (D)
//		if (i + tC < iN && offsetDimension + tD < iD) { // TODO Check speed, maybe unroll last loop
			X[tC][tD] = iData[offsetData + tC * DIM + tD]; // All coalesced reads
//		}

		__syncthreads();

		for(int j=0; j<BLOCK_SIZE_COM; j++) {
			if (I[j] == thisComponent) {
				S[tC][tD] += X[j][tD];
			}
		}
		offsetData += BLOCK_SIZE_COM * DIM;
	}

	// Write results
//	if (thisComponent < iTJ && thisDimension < iD) {
		oMean[thisComponent * DIM + offsetDimension + tD];
//	}
}

// A better approach may involve two kernels:
// 1. Does a sparse partial reduction
// 2. A loop of highly parallelized reductions on the dense results from (1)


#ifdef __cplusplus
}
#endif

#endif /* MVSPARSEAGGREGATE_KERNEL_CU_ */
