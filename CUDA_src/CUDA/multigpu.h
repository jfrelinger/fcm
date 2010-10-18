#if !defined(multi_gpu_h)
	#define multi_gpu_h
	class TGPUplan {
	public:
		TGPUplan(){};
		//Device id
		int device;

		//Host-side input data
		int N;
		REAL *h_X;
		REAL *h_MeanAndSigma;
		REAL *h_Rand;
		INT *h_Component;
		INT *h_Z;
#if defined(TEST_GPU)
		REAL* hDensities;
#endif
#if defined(CDP_MEANCOV)
		REAL *hMean;
		REAL *hCov;
		INT *hIndices;
		INT *hRowIndices;
		INT *hClusterCount;
#endif
		//
		int kT;
		int kJ;
		int kD;
		bool Multithread;
		int NumDevices;

		//device-side variables
		REAL* dX; /** Flattened and padded 2D array to hold the data */
		REAL* dDensity; /** Resultant array to hold density evaluations */
		REAL* dMeanAndSigma; /** Flattened and padded array to hold distribution means, CHDs and logDets */
		REAL* dRandomNumber; /** Array to hold random numbers */
		INT*  dComponent; /** Resultant integer array to hold sampled component IDs */
		INT*  dZ;

#if defined(CDP_MEANCOV)
		REAL* dMean;
		REAL* dCov;
		INT *dIndices;
		INT *dRowIndices;
#endif
	};
	class TGPUEMplan {
	public:
		TGPUEMplan(){};
		//Device id
		int device;

		//Host-side input data
		int N;
		float *h_X;
		float *h_MeanAndSigma;
		float *h_SumPi;
		float *h_Likelihood;
		float *h_pi;
		float *h_piSub;
		float *h_Xbar;
		float *h_shift;
		INT*  h_Z;
		
		//
		int kT;
		int kJ;
		bool Multithread;
		int NumDevices;
		int Chunking;
		
		bool lastIt;

		//device-side variables
		REAL* dX; /** Flattened and padded 2D array to hold the data */
		REAL* dDensity; /** Resultant array to hold density evaluations */
		REAL* dMeanAndSigma; /** Flattened and padded array to hold distribution means, CHDs and logDets */
		REAL* dSumPi;
		REAL* dPartialSum;
		REAL* dLikelihood; /* Resultant vector to hold the log likelihood */ 
		REAL* dXBar;
		REAL* dShift;
		INT*  dZ;
		
		REAL* dRandomNumber;
		REAL* h_Rand;
		int* dComponent;
		int* h_Component;
	};

#endif
