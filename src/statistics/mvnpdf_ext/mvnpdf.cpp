#include "MersenneTwister.h"
#include "newmat.h"
#include "newmatap.h"
#include "specialfunctions2.h"
#include "mvnpdf.h"
#include <math.h>

#include <stdexcept>

#if defined(CDP_CUDA)
#include "cuda_mvnpdf.h"
#define PAD_MULTIPLE  16
#define HALF_WARP  16
#endif

double mvnpdf(int xd, double* px,
	 int md, double* mu,
	 int sd, int sp, double* sigma)
	 {
	 	
		SpecialFunctions2 msf;
		int D = xd;
		SymmetricMatrix Sigma(D);
		LowerTriangularMatrix L;
		LowerTriangularMatrix InvChol;
		for(int i = 0; i<D;++i){
			for(int j=0; j<=i; ++j){
				int pos = i*D+j;
				Sigma(i+1,j+1) = sigma[pos];
			};
		};
		
		L = Cholesky(Sigma);
		
		InvChol = L.i();
		double logdet = msf.logdet(L);
		double val =  msf.mvnormpdf(px, mu, InvChol, D, 0, logdet);
		L.ReleaseAndDelete();
		Sigma.ReleaseAndDelete();
		InvChol.ReleaseAndDelete();
		return val;
	 };

void mvnpdf(int xd, int xp, double* px, 
	 int md, double* mu,
	 int sd, int sp, double* sigma,
	 int outd, double* out)
	 {
	 	for(int i = 0; i<xd; i++){
	 		out[i] = mvnpdf(xp, &px[i*xp], md, mu, sd, sp, sigma);
	 	};
	 };
	 
double wmvnpdf(int xd, double* px,
	 double pi,
	 int md, double* mu,
	 int sd, int sp, double* sigma)
	 {
	 	return pi * mvnpdf(xd, px, md, mu, sd, sp, sigma);
	 };
	 
void wmvnpdf(int xd, int xp, double* px, 
	 double pi,
	 int md, double* mu,
	 int sd, int sp, double* sigma,
	 int outd, double* out)
	 {
	 	for(int i = 0; i<xd; i++){
	 		out[i] = pi * mvnpdf(xp, &px[i*xp], md, mu, sd, sp, sigma);
	 	};
	 };

	 
void wmvnpdf(int xd, int xp, double* px, 
	int pd, double* pi,
	int md, int mp, double*mu,
	int sk, int sd, int sp, double* sigma,
	int outd, double* out)
	{
		if ((xp != mp) || 
		(mp != sd) ||
		(sd != sp) ||
		(pd != md) ||
		(md != sk) ||
		(outd != xd*pd)) {
			throw invalid_argument("objects are not aligned");
		} 
		else {
			int k = pd; // k components
			int n = xd; // n events
			int d = xp; // d dimensions

#if defined(CDP_CUDA)
		cuda_wmvnpdf(n, d, k, px, pi, mu, sigma, out);
#else

			for(int j = 0; j < xd; ++j) {
				for(int i = 0; i< pd; ++i){
					out[pd*j+i] = pi[i] * mvnpdf(xp, &px[j*xp], mp, &mu[mp*i], sd, sp, &sigma[sd*sp*i]);
				};
			};
#endif
		};
	};


#if defined(CDP_CUDA)
void cuda_wmvnpdf(int n, int d, int k,
		double* px, double* pi, double* mu, double* sigma,
		double* out)
{
	int pad = pad_data_dim(n,d);
	int ps = pack_size(k,d);
	float *data = new float[n*pad];
	float *param = new float[ps*k];
        float *nout = new float[n*k];
	pack_param(d, k, mu, sigma, param);
	load_data(n,d,pad,px,data);
	CUDAmvnpdf(data, param, nout, d,n,k,pack_size(k,d), pad);
	for(int i=0; i<n;++i)
	{
		for(int j=0; j<k;++j)
		{
			out[i*k+j] = pi[j] * exp(nout[j*n+i]);
		}
	}
	delete [] data;
	delete [] param;
	delete [] nout;

};

void pack_param(int d, int k, double* mu, double* sigma, float* out)
{
	SpecialFunctions2 msf; // used to find logdet
	int icsize = k * (k + 1) / 2;
	//int pad_stride = k + icsize + 2;
	int pad_stride = pack_size(k,d);
	int sigma_offset = d*d;
	SymmetricMatrix Sigma (d);
	LowerTriangularMatrix L;
	LowerTriangularMatrix InvChol;
	Real *s;
	for (int i=0;i<k;++i)
	{
		// pack mu
		for (int j=0; j<d;++j)
		{
			out[pad_stride*i+j] = mu[j];
		}
		// pack inv chol of sigma...
		for(int l = 0; l<d;++l){
			for(int j=0; j<=l; ++j){
				int pos = l*d+j;
				Sigma.element(l,j) = sigma[pos+(l*sigma_offset)];
			};
		};
		
		L = Cholesky(Sigma);
		
		InvChol =  L.i();
		s = InvChol.data();
		for(int j = 0; j <icsize; ++j){
			out[(pad_stride*i)+k+j] = (float) s[j];
		}
		out[(pad_stride*i)+k+icsize] = 1;
		out[(pad_stride*i)+k+icsize+1] = msf.logdet(L);
		
	}
	L.Release();
	Sigma.Release();
	InvChol.Release();
};

int next_mult(int k, int p) {
	if (k % p) {
		return k + (p - (k % p));
	}
	else
	{
		return k;
	}
};

int pack_size(int n, int d) {
	int k = n*d;
	int icsize = n*(d * ((d + 1) / 2));
	int pad_dim = k + icsize; // # mu + # sigma * size of inv chol of sigma
	pad_dim = next_mult( pad_dim + 2, PAD_MULTIPLE);
	return pad_dim;	
};

int pad_data_dim(int n, int d) {
	if (d % HALF_WARP)
	{
		return d+1;
	}
	else
	{
		return d;
	}
};

void load_data(int n, int d, int pad, double *x, float *out)
{
	for(int i = 0;i<n;++i)
	{
		for(int j = 0; j<d;++j)
		{
			out[i*pad+j] = (float) x[i*d+j];
		}
	}
};


#endif
