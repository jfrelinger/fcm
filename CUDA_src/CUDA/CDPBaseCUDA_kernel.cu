/*
 * CDPBaseCUDA_kernel.cu
 *
 *  Created on: Jul 8, 2009
 *      Author: msuchard
 */


#ifndef _Included_CDPBaseCUDA_kernel
#define _Included_CDPBaseCUDA_kernel

/**************INCLUDES***********/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "CDPBaseCUDA_kernel.h"

/**************CODE***********/
#ifdef __cplusplus
extern "C" {
#endif

cudaError_t gpuMvNormalPDF(
						REAL* iData, /** Data-vector; padded */
						REAL* iDensityInfo, /** Density info; already padded */
						REAL* oMeasure, /** Resultant measure */
						int iD, /** Not currently necessary, as DIM is hardcoded */
						int iN,
						int iTJ
					) {
	// TODO Initialize grid/block variables once
		dim3 gridPDF(iN/DATA_IN_BLOCK, iTJ/DENSITIES_IN_BLOCK);
		if (iN % DATA_IN_BLOCK != 0)
			gridPDF.x += 1;
		if (iTJ % DENSITIES_IN_BLOCK != 0)
			gridPDF.y += 1;
		dim3 blockPDF(DATA_IN_BLOCK,DENSITIES_IN_BLOCK);

	#ifdef DEBUG
		fprintf(stderr,"Grid = (%d,%d);  Block = (%d,%d)\n",gridPDF.x,gridPDF.y,blockPDF.x,blockPDF.y);
		fprintf(stderr,"T = %d\n",kT);
		fprintf(stderr,"DB = %d\n",DENSITIES_IN_BLOCK);
		fprintf(stderr,"DATA_PACK = %d\n",DATA_PADDED_DIM);
		checkCUDAError("Pre-invocation: mvNormalPDF");
	#endif

		mvNormalPDF<<<gridPDF,blockPDF>>>(iData,iDensityInfo,oMeasure,DIM, iN, iTJ);

		return cudaSuccess;
}

cudaError_t gpuSampleFromMeasureMedium(
						REAL* iMeasure, /** Precomputed measure */
						REAL* iRandomNumber, /** Precomputed random number */
						INT* oComponent, /** Resultant choice */
						int iN,
						int iT,
						int iJ
				) {
	// Call GPU - sample
//	dim3 gridSample(kN/SAMPLE_BLOCK,1);
//	if (kN % SAMPLE_BLOCK != 0)
//		gridSample.x +=1 ;
//	dim3 blockSample(SAMPLE_BLOCK,1);
//	sampleFromMeasureSmall<<<gridSample, blockSample>>>(dDensity,dRandomNumber,
//			dComponent,kT * kJ);

	dim3 gridSample(iN/SAMPLE_BLOCK,1);
	if (iN % SAMPLE_BLOCK != 0)
		gridSample.x +=1 ;
	dim3 blockSample(SAMPLE_DENSITY_BLOCK,SAMPLE_BLOCK);
	sampleFromMeasureMedium<<<gridSample, blockSample>>>(iMeasure,iRandomNumber,
			oComponent,iN,iT,iJ);

	return cudaSuccess;
}

cudaError_t gpuMvSparseAggregate(
					REAL* iData, /** Data-vector; padded */
					INT*  iComponent, /** Cluster allocation-vector */
					REAL* oMean, /** Resultant cluster means */
					int iD, /** Not currently necessary, as DIM is hardcoded */
					int iN,
					int iTJ
				) {
#ifdef DEBUG
	fprintf(stderr,"Entering gpuMvSparseAggregate");
#endif


	dim3 gridAggregate (1);
	dim3 blockAggregate (1);
	mvSparseAggregate<<<gridAggregate,blockAggregate>>>(iData,iComponent,oMean, iD, iN, iTJ);

#ifdef DEBUG
	fprintf(stderr,"Exiting gpuMvSparseAggregate");
#endif
}

#ifdef __cplusplus
}
#endif
#endif // _Included_CDPBaseCUDA_kernel
//
