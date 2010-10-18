
#ifndef EM_CDPBASECUDA_H_
#define EM_CDPBASECUDA_H_

#include "CDPBaseCUDA.h"
#if defined(MULTI_GPU)
	#include "GPUWorker.h"
#endif

using namespace std;


class EMCDPBaseCUDA : public CDPBaseCUDA
{
public:
	EMCDPBaseCUDA();
	
	~EMCDPBaseCUDA(void);
	
	int initializeInstanceEM(
					const int iFirstDevice,
					const int iJ,
					const int iT,
					const int iN,
					const int iD,
					const int iNumberDevices,
					const bool keepLastZ
					);
	
	int initializeDataEM(double** iX);
	
	
	void MakeGPUPlansEM(int startdevice, int numdevices);
	
	int uploadMeanAndSigmaEM(
						   RowVector& iP,
						   vector<RowVector>& iMu,
						   //vector<UpperTriangularMatrix>& iSigma,
							vector<LowerTriangularMatrix>& iSigma,
						   vector<double>& iLogDet);
	
	void updatePi(RowVector& iP,
				  vector<RowVector>& Mu,
				  vector<LowerTriangularMatrix>& Sigma,
				  vector<double>& iLogDet,
				  double* SumPi,
				  double* Likelihood);
	
	void updateLastPi(RowVector& iP,
					  vector<RowVector>& Mu,
					  vector<LowerTriangularMatrix>& Sigma,
					  vector<double>& iLogDet,
					  int* Z);
	
	void updateXbar(Matrix& xBar, RowVector& SumPi);
	
	void updateSigma(vector<SymmetricMatrix>& Sigma,
					 RowVector& SumPi,
					 SymmetricMatrix& Phi,
					 Matrix& xBar,
					 double gamma,
					 double nu,
					 RowVector etaEst);
					 
	
	/* Add functions for CUDA EM */
	
	TGPUEMplan plans[MAX_GPU_COUNT];
	CUTThread threadID[MAX_GPU_COUNT];
	
	void finalize(void);
	int isChunking;

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
	REAL* hXBar;
	REAL* hPi;
	REAL* hShift;
	INT* hZ;
	//int kMultipleDevices;
	int kDeviceCount;
	int kStartDevice;
	
	bool kLastZ;
	
	
};

#endif


