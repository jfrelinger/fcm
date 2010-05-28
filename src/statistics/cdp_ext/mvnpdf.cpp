#include "MersenneTwister.h"
#include "newmat.h"
#include "newmatap.h"
#include "specialfunctions2.h"


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
