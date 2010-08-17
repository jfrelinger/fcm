#pragma once
// extension to handle extra EM variables

#if  defined(PYWRAP)
	#include "cdpresult.h"
#endif
#define WANT_STREAM

class CDPResultEM : public CDPResult 
{
public:
	CDPResultEM(int nclusters, int ncomponents, int npoints,int dimension) : CDPResult(nclusters,ncomponents, npoints, dimension){
		postLL = 0.0; pi_ij.ReSize(N,T); sumiPi.ReSize(T); xbar.ReSize(T,D);isEM = 1;}

	~CDPResultEM(void);

	double postLL; //Posterior Log Likelihood (Up to a constant shift)
	
	// Work Variables
	Matrix pi_ij;
	RowVector sumiPi;
	Matrix xbar;
	RowVector etaEst;
	//RowVector maxPdf;
	void savePostLL(string FileName) {
		ofstream theFile(FileName.c_str());
		if (theFile.fail()) {
			std::cout << "Failed to create file " << FileName.c_str()  << endl;
			exit(1);
		}
		theFile << postLL << endl;
		theFile.close();
	}
};
