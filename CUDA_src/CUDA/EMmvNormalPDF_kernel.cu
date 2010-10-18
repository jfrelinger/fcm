/* EMmvNormalPDF_kernel.cu
 *
 * Modification of Marc Suchard's CUDA kernel to calculate a multivariate normal density.
 * This version accepts the transpose of the data matrix X.
 *
 * @author Andrew Cron
 */

#ifndef _Included_mvNormalPDFKernel
#define _Included_mvNormalPDFKernel

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
 */

__global__ void EMmvNormalPDF(
					REAL* inData, /** Data-vector; Transposed */
					REAL* inDensityInfo, /** Density info; already padded */
					REAL* outPDF, /** Resultant PDF */
					int iD, 
					int iN,
					int iTJ,
					int Ntotal
					) {
	const int thidx = threadIdx.x;
	const int thidy = threadIdx.y;

	const int dataBlockIndex = blockIdx.x * DATA_IN_BLOCK;
	const int datumIndex = dataBlockIndex + thidx;

	const int densityBlockIndex = blockIdx.y * DENSITIES_IN_BLOCK;
	const int densityIndex = densityBlockIndex + thidy;

	const int pdfIndex = datumIndex * iTJ + densityIndex;

	extern __shared__ REAL sData[];
	REAL *densityInfo = sData;
	// do this for now, will be more efficient to pass them in as parameters?
	//-------------------------------------------------------
	int LOGDET_OFFSET = iD * (iD + 3) / 2;
	int MEAN_CHD_DIM = iD * (iD + 3) / 2	+ 2;	
	int PACK_DIM = 16;
	while (MEAN_CHD_DIM > PACK_DIM) {PACK_DIM += 16;}
	int DATA_PADDED_DIM = 8;
	//while (iD > DATA_PADDED_DIM) {DATA_PADDED_DIM += 8;}
	//--------------------------------------------------

	const int data_offset = DENSITIES_IN_BLOCK * PACK_DIM;
	REAL *data = &sData[data_offset];
	
	
	// Read in data, ASSUMES: DENSITIES_IN_BLOCK >= DIM
	//if (thidy < iD && datumIndex < iN ) {
	//	data[thidx * iD + thidy] = inData[thidy*Ntotal + datumIndex];
	//}									
		
	for(int chunk=0; chunk<iD; chunk+=DENSITIES_IN_BLOCK){
		if (thidy + chunk < iD && datumIndex < iN) {
				data[thidx * iD + thidy + chunk] = inData[(thidy+chunk)*Ntotal + datumIndex];
		}
	}

	// Read in density info by chunks
	for(int chunk = 0; chunk < PACK_DIM; chunk += DATA_IN_BLOCK) {
		if (chunk + thidx < PACK_DIM) {
			densityInfo[thidy * PACK_DIM + chunk + thidx] = inDensityInfo[PACK_DIM*densityIndex	+ chunk + thidx];
		}
	}

	__syncthreads();

	// Setup pointers
	
	REAL* tData = data+thidx*iD;
	REAL* tDensityInfo = densityInfo + thidy * PACK_DIM;
	
	
	REAL* tMean = tDensityInfo;			//do we need to unallocate shared/register variables?
	REAL* tSigma = tDensityInfo + iD;
	REAL  tP = tDensityInfo[LOGDET_OFFSET];
	REAL  tLogDet = tDensityInfo[LOGDET_OFFSET+1];
	
	// Do density calculation
	REAL discrim = 0;
	for(int i=0; i<iD; i++) {
		REAL sum = 0;
		for(int j=0; j<=i; j++) {
			sum += *tSigma++ * (tData[j] - tMean[j]); // xx[j] is always calculated since j <= i
		}
		discrim += sum * sum;
	}
	REAL mydim = (REAL)iD;
    //REAL d = exp(log(tP) -0.5 * (discrim + tLogDet + (LOG_2_PI*mydim)));
	//REAL d = tP * exp(-0.5 * (discrim + tLogDet + (LOG_2_PI*mydim)));

	REAL d = log(tP) - 0.5 * (discrim + tLogDet + (LOG_2_PI*mydim));

	if (datumIndex < iN & densityIndex < iTJ)
		outPDF[pdfIndex] = d;
}

cudaError_t EMgpuMvNormalPDF(
					REAL* inData, /** Data-vector; padded */
					REAL* inDensityInfo, /** Density info; already padded */
					REAL* outPDF, /** Resultant PDF */
					int iD, 
					int iN,
					int iTJ,
					int Ntotal
					) {
		dim3 gridPDF(iN/DATA_IN_BLOCK, iTJ/DENSITIES_IN_BLOCK);
		if (iN % DATA_IN_BLOCK != 0)
			gridPDF.x += 1;
		if (iTJ % DENSITIES_IN_BLOCK != 0)
			gridPDF.y += 1;
		dim3 blockPDF(DATA_IN_BLOCK,DENSITIES_IN_BLOCK);
		int sharedMemSize = (DENSITIES_IN_BLOCK * PACK_DIM + DATA_IN_BLOCK * DIM) * SIZE_REAL;
		EMmvNormalPDF<<<gridPDF,blockPDF,sharedMemSize>>>(inData,inDensityInfo,outPDF,iD, iN, iTJ, Ntotal);
		cudaThreadSynchronize(); 
		return cudaSuccess;
}

#ifdef __cplusplus
}
#endif
#endif // _Included_mvNormalPDFKernel


