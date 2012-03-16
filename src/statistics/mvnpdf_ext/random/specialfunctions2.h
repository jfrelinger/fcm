#pragma once
#include "SpecialFunctions.h"
class LowerTriangularMatrix;
class UpperTriangularMatrix;
class SymmetricMatrix;
class RowVector;

class SpecialFunctions2 :
	public SpecialFunctions
{
public:
	SpecialFunctions2(void);
	~SpecialFunctions2(void);
	static double mvtpdf(double* x, double* mu, UpperTriangularMatrix& Sigma, double nu, int dim, int logspace,double logdet,double precalculate);
	static double mvtpdf(double* x, double* mu, LowerTriangularMatrix& Sigma, double nu, int dim, int logspace,double logdet,double precalculate);
	static double mvnormpdf(double* x, double* mu, LowerTriangularMatrix& Sigma, int dim, int logspace,double logdet);
	static double mvnormpdf(double* x, double* mu, UpperTriangularMatrix& Sigma, int dim, int logspace,double logdet);
	static double mvnormpdf_unwind2(double* x, double* mu, double* s,double logdet);
	static double mvnormpdf_unwind2_weighted(double* x, double* para);
	
	static double invwishartpdf(LowerTriangularMatrix& Xinvchol,double logdetX, double s, SymmetricMatrix& S,double logdetS, int dim, int logspace);

	static SymmetricMatrix invwishartrand(int nu, LowerTriangularMatrix& Sinvchol,MTRand& mt);
	static SymmetricMatrix wishartrand(int nu, LowerTriangularMatrix& Sinvchol,MTRand& mt);
	static RowVector mvnormrand(RowVector& mu, LowerTriangularMatrix& cov,MTRand& mt);
	static RowVector mvnormrand(RowVector& mu, SymmetricMatrix& cov,MTRand& mt);
	double logdet(LowerTriangularMatrix& lchol);
};
