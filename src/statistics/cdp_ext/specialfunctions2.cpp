#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "MersenneTwister.h"
#include "specialfunctions2.h"
#define LOG_2_PI 1.83787706640935
#define LOG_PI 1.144729885849400
SpecialFunctions2::SpecialFunctions2(void)
{
}
SpecialFunctions2::~SpecialFunctions2(void)
{
}
double SpecialFunctions2::logdet(LowerTriangularMatrix& lchol) {
	double log_det_sigma = 0;
	int dim = lchol.Ncols();
	for (int i = 0; i < dim; i++) {
		log_det_sigma += log(lchol[i][i]);
	}
	log_det_sigma *= 2.0;
	return log_det_sigma;
}
double SpecialFunctions2::mvtpdf(double* x, double* mu,UpperTriangularMatrix& Sigma, double nu, int dim, int logspace,double logdet,double precalculate) {
	double discrim,mydim;
	int i,j;
	mydim = (double) dim;

	double* xx = new double[dim];
	for (i = 0; i < dim; i++) {
		xx[i] = x[i] - mu[i];
	}
	discrim = 0;
	double* s = Sigma.Store();
	for ( i = 0; i < dim; i++) {
		double sum = 0;
		for (j = i; j < dim; j++) {
			sum+= *s++ * xx[j];
		}
		discrim+= sum * sum;
	}
	delete [] xx;

	double d = precalculate - 0.5 * logdet  -0.5 * (nu + mydim) * log(1+discrim / nu);
	if (!logspace) {
		d = exp(d);
	}
	return d;
}
double SpecialFunctions2::mvnormpdf(double* x, double* mu, UpperTriangularMatrix& Sigma, int dim, int logspace,double logdet) {
	double discrim;
	int i,j;
	double* xx = new double[dim];
	for (i = 0; i < dim; i++) {
		xx[i] = x[i] - mu[i];
	}
	discrim = 0;
	double* s = Sigma.Store();
	for ( i = 0; i < dim; i++) {
		double sum = 0;
		for (j = i; j < dim; j++) {
			sum+= *s++ * xx[j];
		}
		discrim+= sum * sum;
	}
	delete [] xx;
    double d = -0.5 * (discrim + logdet + (dim*LOG_2_PI));
    if (!logspace) {
		d = exp(d);
	}
	return d;
}
SymmetricMatrix SpecialFunctions2::invwishartrand(int nu, LowerTriangularMatrix& Sinvchol,MTRand& mt) {
	// get back the original degrees of freedom
	int i ,j;
	int dim = Sinvchol.Ncols();
	nu = nu+dim - 1;
	LowerTriangularMatrix foo(dim); 
	double* f = foo.Store();
	for (i = 0; i < dim; i++) {
		for (j = 0; j < i; j++) {
			*f++ = mt.randNorm(0,1); 
		}
		*f++ = sqrt(chi2rand(nu-i,mt)); 
	}
	//LowerTriangularMatrix blah = (Sinvchol * foo).i();
	
	LowerTriangularMatrix blah(dim);
	double *b = blah.Store();
	for (i=0; i < dim; i++) {
		double *s = Sinvchol[i];
		for (j = 0; j <= i; j++) {
			double sum = 0;
			for (int k = j; k<=i; k++) {
				sum+= s[k] * foo[k][j];
			}
			*b++ = sum;
		}
	}
	blah << blah.i();

	SymmetricMatrix r(dim);
	r << blah.t() * blah;
	return r;
}
SymmetricMatrix SpecialFunctions2::wishartrand(int nu, LowerTriangularMatrix& Sinvchol,MTRand& mt) {
	int dim = Sinvchol.Ncols();
	
	int i,j;
	LowerTriangularMatrix foo(dim); 
	double* f = foo.Store();
	for (i = 0; i < dim; i++) {
		for (j = 0; j < i; j++) {
			*f++ = mt.randNorm(0,1); 
		}
		*f++ = sqrt(chi2rand(nu-i,mt)); 
	}

	//LowerTriangularMatrix blah = Sinvchol * foo;
	LowerTriangularMatrix blah(dim);
	double *b = blah.Store();
	for (i=0; i < dim; i++) {
		double *s = Sinvchol[i];
		for (j = 0; j <= i; j++) {
			double sum = 0;
			for (int k = j; k<=i; k++) {
				sum+= s[k] * foo[k][j];
			}
			*b++ = sum;
		}
	}

	SymmetricMatrix r(dim); 
	r << blah * blah.t();
    return r;
}

RowVector SpecialFunctions2::mvnormrand(RowVector& mu, LowerTriangularMatrix& cov,MTRand& mt){
	int dim = cov.Ncols();
	double * rn = new double[dim];
	RowVector r(dim);
	double *s = r.Store();
	double *c = cov.Store();
	double *m = mu.Store();
	for ( int i = 0; i < dim; i++) {
		double sum = 0;
		rn[i] = mt.randNorm(0,1);
		for (int j = 0; j <=i; j++) {
			sum += *c++ * rn[j]; 
		}
		*s++ = *m++ + sum ; 
	}
	delete [] rn;
	return r;
}

RowVector SpecialFunctions2::mvnormrand(RowVector& mu, SymmetricMatrix& cov,MTRand& mt) {
	LowerTriangularMatrix ltcov = Cholesky(cov);
	return mvnormrand(mu,ltcov,mt);
}
