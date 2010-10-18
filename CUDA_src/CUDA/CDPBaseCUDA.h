/*
 * CDPBaseCUDA.h
 *
 *  Created on: Jun 29, 2009
 *      Author: msuchard
 */

#ifndef CDPBASECUDA_H_
#define CDPBASECUDA_H_

#include "newmatap.h"
#include "MersenneTwister.h"
#include "CUDASharedFunctions.h"
#if defined(MULTI_GPU)
	#include "GPUWorker.h"
#endif

#include <vector>


#if _WIN32
	#undef INT				//need to get rid of the INT define in CUDASharedFunctions.h
#endif
#include <multithreading.h>
#include "multigpu.h"
#include "stdafx.h"

class CDPBaseCUDA {
	public:
		CDPBaseCUDA();

		~CDPBaseCUDA(void);

		int initializeInstance(
				const int iFirstDevice,
				const int iJ,
				const int iT,
				const int iN,
				const int iD,
				const int iNumberDevices
				);

		int initializeData(double** iX);

		int uploadMeanAndSigma(
				RowVector& iP,
				vector<RowVector>& iMu,
				vector<LowerTriangularMatrix>& iSigma,
				vector<double>& iLogDet);

		int uploadRandomNumbers(MTRand& mt);

		void MakeGPUPlans(int startdevice, int numdevices);
		int sampleWK(
				RowVector& iQ,
				vector<RowVector>& iP,
				vector<RowVector>& iMu,
				vector<LowerTriangularMatrix>& iSigma,
				vector<double>& iLogDet,
				MTRand& mt,
				int* oW,
				int* oK,
				int* oZ
				);

		TGPUplan      plans[MAX_GPU_COUNT];
		CUTThread threadID[MAX_GPU_COUNT];
		
		int getGPUDeviceCount(void);

		void printGPUInfo(int iDevice);

		void getGPUInfo(int iDevice, char *oName, int *oMemory, int *oSpeed);

	private:

		void finalize(void);

		int initializedInstance;
		int kJ;
		int kT;
		int kN;
		int kD;

		#if defined(MULTI_GPU)
			GPUWorker** workers;
		#endif

		REAL* hX;
		REAL* hMeanAndSigma;
		INT*  hComponent;
		REAL* hRandomNumber;
		INT*  hZ;
#if defined(TEST_GPU)
		REAL* hDensities;
#endif
		//int kMultipleDevices;
		int kDeviceCount;
		int kStartDevice;
#if defined(CDP_MEANCOV)
		INT* hIndices;
		INT* hRowIndices;
#endif

public:
#if defined(CDP_MEANCOV)
	REAL* hMean;
	REAL* hCov;
	INT* hClusterCount;
#endif

};

#endif /* CDPBASECUDA_H_ */
