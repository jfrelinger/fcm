#ifndef _MVSPAGGREGATE_KERNEL_H_
#define _MVSPAGGREGATE_KERNEL_H_
//#include "stdafx.h"
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "CUDASharedFunctions.h"
void csr(int *indata, int numRows, int size, 
		int *rowIndices, /** T + 1 length */
		int *indices,
		int *clustercount) {
	
	int i;
	int *currentCount = new int[numRows];
	for (i = 0; i < numRows; i++)  {
		rowIndices[i] =0;
		currentCount[i] = 0;
	}	
	for (i = 0; i < size; i++) {
		rowIndices[indata[i]]++;
	}
	int cumCount = rowIndices[0];
	clustercount[0] = rowIndices[0];
	rowIndices[0] = 0;
	for (i = 1; i < numRows; i++) {
		int ntemp = rowIndices[i];
		rowIndices[i] = cumCount;
		cumCount += ntemp;
		clustercount[i] = ntemp;
	}
	rowIndices[numRows] = size;
	for (i = 0;  i <size; i++) {
		int row = indata[i];
		indices[rowIndices[row]+currentCount[row]] = i;
		currentCount[row]++;
	}
	delete [] currentCount;
	currentCount = NULL;
}
__global__ void mvSparseAggregateCrossProduct(
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
				) {
	
	unsigned int dim = threadIdx.x;

	unsigned int tid = threadIdx.y;
	unsigned int bid = blockIdx.y;
	
	unsigned int ind2Dx = tid & (HALFWARP-1);
	unsigned int ind2Dy = tid >> HALFWARP_LOG2;
	
	unsigned int ub, lb;
	unsigned int myblock = bid * (BLOCK_SIZE_ROW/HALFWARP);
	unsigned int myi = myblock + ind2Dy;
	
	__shared__ int rowInd[(BLOCK_SIZE_ROW/HALFWARP)+1];
	__shared__ float tempProd[BLOCK_SIZE_COL*(BLOCK_SIZE_ROW/HALFWARP)][HALFWARP+PAD];
	
#ifdef READ_ONCE
	__shared__ float vals[BLOCK_SIZE_ROW][BLOCK_SIZE_COL];
#endif
	
	if ((tid <= ((BLOCK_SIZE_ROW/HALFWARP))) && (myi < numRows) && dim == 0)
		rowInd[tid] = rowIndices[myblock + tid];
	
	__syncthreads();
	
	float t = 0;
	lb = rowInd[ind2Dy] + ind2Dx;
	ub = rowInd[ind2Dy + 1];
	
	if (myi < numRows) {
		for (unsigned int j = lb; j < ub; j += HALFWARP) {
			int ind = indices[j]; // All coalesced memory reads
			if (ind >= 0) {
#ifdef READ_ONCE
				vals[tid][dim] = y[ind * packeddim +dim];
				__syncthreads();
				t += vals[tid][dim] * vals[tid][crossIndex];
#else // Read twice
				float yval = y[ind * packeddim + dim];  // Reads all BLOCK_SIZE_COL entries at once
				float crossval = y[ind * packeddim + crossIndex]; // Should be a single read and broadcast
				t += yval * crossval;
#endif // READ_ONCE				
			}
		}
		tempProd[ind2Dy*BLOCK_SIZE_COL + dim][ind2Dx] = t;
	}
	
	__syncthreads();

	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows) && dim < numColsInput) {
		t = tempProd[ind2Dy*BLOCK_SIZE_COL+dim][0]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][1] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][2]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][3]  
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][4]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][5]  
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][6]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][7]  
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][8]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][9]   
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][10] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][11] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][12] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][13] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][14] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][15];
		if (dim >= crossIndex ) {
			x[myi*numColsOutput + extraOffset + dim] = t;
		}
	}
}
   
__global__ void mvSparseAggregate2(
				float *x, 
				const int *rowIndices, 
				const int *indices,
				const float *y, 
				const unsigned int numRows, 
				const unsigned int numCols,
				const unsigned int packeddim
				) {
	
	unsigned int dim = threadIdx.x;
	
	unsigned int tid = threadIdx.y;
	unsigned int bid = blockIdx.y;
	
	unsigned int ind2Dx = tid & (HALFWARP-1);
	unsigned int ind2Dy = tid >> HALFWARP_LOG2;
	
	unsigned int ub, lb;
	unsigned int myblock = bid * (BLOCK_SIZE_ROW/HALFWARP);
	unsigned int myi = myblock + ind2Dy;
	
	__shared__ int rowInd[(BLOCK_SIZE_ROW/HALFWARP)+1];
//	__shared__ float tempProd[(BLOCK_SIZE_ROW/HALFWARP)][(HALFWARP+PAD)*BLOCK_SIZE_COL];
	__shared__ float tempProd[BLOCK_SIZE_COL*(BLOCK_SIZE_ROW/HALFWARP)][HALFWARP+PAD];	
	
	if ((tid <= ((BLOCK_SIZE_ROW/HALFWARP))) && (myi < numRows) && dim == 0)
		rowInd[tid] = rowIndices[myblock + tid];
	
	__syncthreads();
	
	float t = 0;
	lb = rowInd[ind2Dy] + ind2Dx;
	ub = rowInd[ind2Dy + 1];
	
	if (myi < numRows) {
		for (unsigned int j = lb; j < ub; j += HALFWARP) {
			int ind = indices[j]; // All coalesced memory reads
			if (ind >= 0) {
#if CACHE
				float yval = tex1Dfetch(tex_y_float, ind);
#else
				float yval = y[ind * packeddim + dim];  // Reads all BLOCK_SIZE_COL entries at once
				// TODO Store into shared memory and compute cross-products here
#endif
				t += yval; // Notice no multiplication by 1.0, as in original SpMV
			}
		}
//		tempProd[ind2Dy][(ind2Dx)*BLOCK_SIZE_COL+dim] = t;
		tempProd[ind2Dy*BLOCK_SIZE_COL + dim][ind2Dx] = t;
	}
	
	__syncthreads();

#if 1
	// Works for HALFWARP=16
	if ((ind2Dx == 0) && (myi < numRows) && dim < numCols) {
		t = tempProd[ind2Dy*BLOCK_SIZE_COL+dim][0]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][1] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][2]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][3]  
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][4]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][5]  
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][6]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][7]  
          + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][8]  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][9]   
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][10] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][11] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][12] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][13] 
		  + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][14] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][15];
		x[myi*numCols + dim] = t;
	}
#endif
#if 0
	if (myi < numRows) {	
		if (ind2Dx < 8) 
			tempProd[ind2Dy*BLOCK_SIZE_COL+dim][ind2Dx] += tempProd[ind2Dy*BLOCK_SIZE_COL+dim][(ind2Dx+8)];
		if (ind2Dx < 4) 
			tempProd[ind2Dy*BLOCK_SIZE_COL+dim][ind2Dx] += tempProd[ind2Dy*BLOCK_SIZE_COL+dim][(ind2Dx+4)];
		if (ind2Dx < 2) 
			tempProd[ind2Dy*BLOCK_SIZE_COL+dim][ind2Dx] += tempProd[ind2Dy*BLOCK_SIZE_COL+dim][(ind2Dx+2)];
		if (ind2Dx < 1 && dim < numCols) 
			x[myi*numCols + dim]= tempProd[ind2Dy*BLOCK_SIZE_COL+dim][ind2Dx] + tempProd[ind2Dy*BLOCK_SIZE_COL+dim][(ind2Dx+1)];
	}
	__syncthreads();
#endif
}

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
					) {
	csr(iLabels,iT,iN,h_rowIndices,h_indices, h_clusterCount);
	// Copy onto device 
	cutilSafeCall( cudaMemcpy(d_rowIndices, h_rowIndices, sizeof(int) * (iT + 1),
	                                cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(d_indices, h_indices, sizeof(int) * iN,
		                                cudaMemcpyHostToDevice) );
	cudaThreadSynchronize();	
	
    unsigned int gridParam;
    gridParam = (unsigned int) floor((float)iT/(BLOCK_SIZE_ROW/HALFWARP));
    if ((gridParam * (BLOCK_SIZE_ROW/HALFWARP)) < iT) gridParam++;
    dim3 grid2(1, gridParam);
    dim3 block2(BLOCK_SIZE_COL, BLOCK_SIZE_ROW);
    cutilSafeCall( cudaMemset(oMean,0,sizeof(float) * iT * iD));
    mvSparseAggregate2 <<<grid2, block2>>> (
    		oMean,
    		d_rowIndices, 
    		d_indices, 
    		d_iData, 
    		iT, 
    		iD,
			iPackedD);
    cudaThreadSynchronize();
	
	int offset = 0;
	unsigned int crossD = iD * iD;
	cutilSafeCall( cudaMemset(oCov,0,sizeof(float) * iT * iD * iD));
	for(int i = 0; i < iD; i++) { // Loop of each dim:
								 // 1st:  (0,...,D) x 0
								 // 2nd:  (1,...,D) x 1
								 // 
								 // Lst:  D x D
		mvSparseAggregateCrossProduct <<<grid2, block2>>> (
				oCov,
				d_rowIndices,
				d_indices,
				d_iData,
				iT,
				iD,
				i, // Compute products against i-th index
				crossD,
				offset,
				iPackedD);
		offset += iD - i - 1;
	}
	cudaThreadSynchronize();
	
	return cudaSuccess;
}

// Fills oIndices with (0,1,\ldots,N-1)
__global__ void fillSequentially(int* oIndices, const unsigned int N) {
	
	unsigned int tid = blockIdx.x * COMPACT_BLOCK + threadIdx.x;
	
	if (tid < N) {
		oIndices[tid] = tid;
	}	
}

// Fills trailing (from J+1 end) 0s with N.  These 0s results when clusters J-1, J-2, etc. have no components
__global__ void fillInZeros(unsigned int* rowIndices,
							const unsigned int N,
							const unsigned int J) {
	int i = 1;
	while( rowIndices[J-i] == 0) {
		rowIndices[J - i] = N;
		i++;
	}
}

#endif // _MVSPAGGREGATE_KERNEL_H_

