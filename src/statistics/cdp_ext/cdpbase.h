#pragma once

#include "specialfunctions2.h"
#include "cdpprior.h"

class CDPBase
{
public:
	CDPBase();
	~CDPBase(void);

	#if defined(CDP_TBB)
	int sampleW(RowVector& x, RowVector& q, concurrent_vector<RowVector>&p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, concurrent_vector<double>& logdet, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, concurrent_vector<double>& logdet, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, int index,concurrent_vector<double>& logdet, MTRand& mt);

	#else
	int sampleW(RowVector& x, RowVector& q, vector<RowVector>& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, vector<double>& loget, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, vector<double>& logdet, MTRand& mt);

	int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, int index,vector<double>& logdet, MTRand& mt);
	#endif
	
	int sample(double* w, int n, MTRand& mt);
	void cov(Matrix& x, RowVector& mu,int mul, SymmetricMatrix& result);
	double sampleAlpha(double* V, double e, double f, MTRand& mt);
	void sampleMuSigma(Matrix& x, int n, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu, SymmetricMatrix& PostSigma,UpperTriangularMatrix& uti, double& logdet, MTRand& mt);
	void sampleMuSigma(vector<int>& indexes, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu,SymmetricMatrix& PostSigma,UpperTriangularMatrix& uti, double& logdet, MTRand& mt);
	void sampleP(vector<int>& p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt); 
	void sampleP(int* p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt); 
	void sampleM(int n, double gamma, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma, RowVector& m0, 
		SymmetricMatrix& Phi0, RowVector& newm, MTRand& mt);
	void samplePhi(int n, double nu, vector<SymmetricMatrix>& Sigma, double nu0, SymmetricMatrix& Lambda0, 
		SymmetricMatrix& newphi, MTRand& mt);
	//MTRand mt;
	SpecialFunctions2 msf;
	//Matrix mX;
	double** mX;
	CDPPrior prior;
	double precalculate;

	//double sampleWj(RowVector& x, double q, RowVector& m, SymmetricMatrix& Phi, double nu, double gamma);
	//int sampleW(RowVector& x, RowVector& q, vector<RowVector>& m, vector<SymmetricMatrix>& Phi, double nu, double gamma); 
	//int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma);
	//int sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma, int index);
};
