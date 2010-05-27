#include "MersenneTwister.h"
#include "newmat.h"
#include "specialfunctions2.h"


double mvnpdf(int xd, double* px,
	 int md, double* mu,
	 int sd, int sp, double* sigma)
	 {
		SpecialFunctions2 msf;
		int D = xd;
		UpperTriangularMatrix Sigma(D);
		for(int i = 0; i<D;++i){
			for(int j=i; j<D; ++j){
				int pos = i*D+j;
				Sigma[i][j] = sigma[pos];
			};
		};
		LogAndSign las = Sigma.LogDeterminant();
		double logdet = las.Sign() *las.LogValue();
		return msf.mvnormpdf(px, mu, Sigma, D, 0, logdet);
	 };
