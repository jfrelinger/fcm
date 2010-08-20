#include "MersenneTwister.h"
#include "newmat.h"
#include "newmatap.h"
#include "specialfunctions2.h"
#include "mvnpdf.h"

#include <stdexcept>

#if defined(CDP_CUDA)
#include "CDPBaseCUDA.h"
#include "CDPBaseCUDA_kernel.h"
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
		if ((xp != mp) or 
		(mp != sd) or
		(sd != sp) or
		(pd != md) or
		(md != sk) or
		(outd != xd*pd)) {
			throw invalid_argument("objects are not aligned");
		} 
		else {
#if defined(CDP_CUDA)
		
#endif
			for(int j = 0; j < xd; ++j) {
				for(int i = 0; i< pd; ++i){
					out[pd*j+i] = pi[i] * mvnpdf(xp, &px[j*xp], mp, &mu[mp*i], sd, sp, &sigma[sd*sp*i]);
				};
			};

		};
	};

