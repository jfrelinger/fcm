/* cdpbase.h
 * @author Quanli Wang, quanli@stat.duke.edu
 */

#pragma once
#if defined(CDP_CUDA)
	#include "CDPBaseCUDA.h"
	#include "EMCDPBaseCUDA.h"
#endif

#if defined(PYWRAP)
#include "specialfunctions2.h"
#include "cdpprior.h"
#endif

class CDPBase
{
public:
	CDPBase();
	~CDPBase(void);

	#if defined(CDP_TBB)
	int sampleW(RowVector& x, RowVector& q, concurrent_vector<RowVector>&p, concurrent_vector<RowVector>& mu, 
		concurrent_vector<LowerTriangularMatrix>& L_i, concurrent_vector<double>& logdet, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<LowerTriangularMatrix>& L_i, 
		concurrent_vector<double>& logdet, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<LowerTriangularMatrix>& L_i, 
		int index,concurrent_vector<double>& logdet, MTRand& mt);
	
	int sampleK(RowVector& x, int* Z, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<LowerTriangularMatrix>& L_i, 
		concurrent_vector<double>& logdet, MTRand& mt); //Returns classification max Z.
	int sampleK(RowVector& x, int* Z, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<LowerTriangularMatrix>& L_i, 
		int index,concurrent_vector<double>& logdet, MTRand& mt);
	

	void sampleWK(RowVector& x, RowVector& q, concurrent_vector<RowVector>& p, concurrent_vector<RowVector>& mu, 
		concurrent_vector<LowerTriangularMatrix>& L_i, concurrent_vector<double>& logdet, int& neww, int& newk, MTRand& mt);

	#else
	int sampleW(RowVector& x, RowVector& q, vector<RowVector>& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, vector<double>& loget, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, vector<double>& logdet, MTRand& mt);
	int sampleK(RowVector& x, int* Z, RowVector& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, vector<double>& logdet, MTRand& mt);
	int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, int index,vector<double>& logdet, MTRand& mt);
	int sampleK(RowVector& x, int* Z, RowVector& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, int index,vector<double>& logdet, MTRand& mt);
	
	void sampleWK(RowVector& x, RowVector& q, vector<RowVector>& p, vector<RowVector>& mu, vector<LowerTriangularMatrix>& L_i, 
		vector<double>& logdet, int& neww, int& newk, MTRand& mt);
	#endif

	
	int sample(double* w, int n, MTRand& mt);
	void cov(Matrix& x, RowVector& mu,int mul, SymmetricMatrix& result);
	double sampleAlpha(double* V, double e, double f, MTRand& mt);
	void sampleMuSigma(Matrix& x, int n, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu, SymmetricMatrix& PostSigma,LowerTriangularMatrix& li, double& logdet, MTRand& mt);
	void sampleMuSigma(vector<int>& indexes, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu,SymmetricMatrix& PostSigma,LowerTriangularMatrix& li, double& logdet, MTRand& mt);
	void sampleP(vector<int>& p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt);
	void sampleP(int* p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt);
	void samplePhi(int n, double nu, vector<SymmetricMatrix>& Sigma, double nu0, SymmetricMatrix& Lambda0,
		SymmetricMatrix& newphi, MTRand& mt);

	SpecialFunctions2 msf;
	double** mX;
	CDPPrior prior;
	double precalculate;
	
#ifdef CDP_CUDA
	CDPBaseCUDA cuda;
	EMCDPBaseCUDA emcuda;
#endif

};
