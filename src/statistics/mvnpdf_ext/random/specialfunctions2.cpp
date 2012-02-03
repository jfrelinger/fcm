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
double SpecialFunctions2::mvtpdf(double* x, double* mu,LowerTriangularMatrix& Sigma, double nu, int dim, int logspace,double logdet,double precalculate) {
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
		for (j = 0; j <= i; j++) {
			sum+= *s++ * xx[j];
		}
		discrim+= sum * sum;
	}
	delete [] xx;
	xx = NULL;

	double d = precalculate - 0.5 * logdet  -0.5 * (nu + mydim) * log(1+discrim / nu);
	if (!logspace) {
		d = exp(d);
	}
	return d;
}

double SpecialFunctions2::mvtpdf(double* x, double* mu,UpperTriangularMatrix& Sigma, double nu, int dim, int logspace,double logdet,double precalculate) {
	LowerTriangularMatrix L = Sigma.t();
	return mvtpdf(x,mu,L,nu,dim,logspace,logdet,precalculate);
}
double SpecialFunctions2::mvnormpdf_unwind2(double* x, double* mu, double* s,double logdet) {
	double sum0 = s[0] * (x[0]-mu[0]);
	double sum1 = s[1] * (x[0]-mu[0]) + s[2] * (x[1]-mu[1]);
	return exp(-0.5 * (sum0 * sum0 + sum1 * sum1 + logdet) - LOG_2_PI);
}

double SpecialFunctions2::mvnormpdf_unwind2_weighted(double* x, double* para) {
	double sum0 = para[3] * (x[0]-para[1]);
	double sum1 = para[4] * (x[0]-para[1]) + para[5] * (x[1]-para[2]);
	return para[0] * exp(-0.5 * (sum0 * sum0 + sum1 * sum1 + para[6]) - LOG_2_PI);
}

// requires the inverse of the lower triangular cholesky factor of the covariance
double SpecialFunctions2::mvnormpdf(double* x, double* mu, LowerTriangularMatrix& L_i, int dim, int logspace,double logdet) {
	double discrim;
	int i,j;
	double mydim = (double) dim;
	double* xx = new double[dim];
	for (i = 0; i < dim; i++) {
		xx[i] = x[i] - mu[i];
	}
	discrim = 0;
	double* s = L_i.Store();
	for ( i = 0; i < dim; i++) {
		double sum = 0;
		for (j = 0; j <= i; j++) {
			sum+= *s++ * xx[j];
		}
		discrim+= sum * sum;
	}
	//	cout << "discrim = " << discrim << "\n";
	delete [] xx;
	xx = NULL;
    double d = -0.5 * (discrim + logdet + mydim * LOG_2_PI);
    if (!logspace) {
		d = exp(d);
	}
	return d;
}

// requires the inverse of the upper triangular cholesky factor of the covariance
double SpecialFunctions2::mvnormpdf(double* x, double* mu, UpperTriangularMatrix& Sigma_t_i, int dim, int logspace,double logdet) {
	LowerTriangularMatrix L = Sigma_t_i.t();
	return mvnormpdf(x,mu,L,dim,logspace,logdet);
}


// WARNING: evaluates the non-normalized inverse wishart pdf -- does
// not include the 2^dp pi^{p(p-1)/4}... terms this is fine so long as
// these values are only used in ratios with one another and have the
// same degrees of freedom and the same size.
double SpecialFunctions2::invwishartpdf(LowerTriangularMatrix& Xinvchol,double logdetX, double s, SymmetricMatrix& S,double logdetS, int dim, int logspace)
{
  double d=0;
  d = (double) (s+dim-1) / 2.0 * logdetS - ((double) dim + s/2.0)*logdetX;
  std::cout << "invwishartpdf::dobule check this value" << endl;
  //  cout << "d = " << d << "\n";
  Matrix temp = Xinvchol.t()*S*Xinvchol;
  
  for(int i=0;i<dim;i++)
    d-=temp[i][i]/2.0;
  
  if(!logspace)
    d = exp(d);
  return d;

}

// here nu = s = d-p+1, and Sinvchol = Chol(S^-1) = Chol(A).  Produces draws with E(X) = S/(s-2)
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

// here nu = d and Sinvchol = Chol(A).  Produces draws with E(X) = dA
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
			double sum = 0.0;
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
		double sum = 0.0;
		rn[i] = mt.randNorm(0,1);
		for (int j = 0; j <=i; j++) {
			sum += *c++ * rn[j]; 
		}
		*s++ = *m++ + sum ; 
	}
	delete [] rn;
	rn = NULL;
	return r;
}

RowVector SpecialFunctions2::mvnormrand(RowVector& mu, SymmetricMatrix& cov,MTRand& mt) {
	LowerTriangularMatrix ltcov = Cholesky(cov);
	return mvnormrand(mu,ltcov,mt);
}
