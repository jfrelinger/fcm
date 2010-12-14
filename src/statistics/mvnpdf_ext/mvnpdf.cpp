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
			int k = pd; // k components
			int n = xd; // n events
			int d = xp; // d dimensions
/*
#if defined(CDP_CUDA)
		cuda_wmvnpdf(n, d, k, px, pi, mu, sigma, out);
#else
*/
			for(int j = 0; j < xd; ++j) {
				for(int i = 0; i< pd; ++i){
					out[pd*j+i] = pi[i] * mvnpdf(xp, &px[j*xp], mp, &mu[mp*i], sd, sp, &sigma[sd*sp*i]);
				};
			};
//#endif
		};
	};


#if defined(CDP_CUDA)
void cuda_wmvnpdf(int n, int d, int k,
		double* px, double* pi, double* mu, double* sigma,
		double* out)
{
	int outd = k*n;
	//DIM = d;
	// load data
	REAL* cx = cuda_load_data(n, d, px);
	// load pi mu sigma
	REAL* param = cuda_load_param(k, d, pi, mu, sigma);
	// call cuda_pdf
	REAL* result;
	REAL* tmp= new REAL[outd];
//	for (int i=0;i<outd;i++)
//	{
//		tmp[i] = 0;
//	}
	cudaMalloc( (void**)&result, n*k*SIZE_REAL);
	gpuMvNormalPDF(cx, param, result, n, d, k);
	cudaMemcpy(tmp, result, n*k*SIZE_REAL, cudaMemcpyDeviceToHost);
	for (int i=0;i<n*k;i++)
	{
		out[i] = (double) tmp[i];
		//std::cout << out[i] << " ";
		//std::cout << tmp[i] << " ";
	}
	std::cout << std::endl;
	// cleanup
	delete [] tmp;
	cudaFree(cx);
	cudaFree(param);
	cudaFree(result);

};

REAL * cuda_load_data(int n, int d, double* px)
{
	REAL * hx = new REAL[n*d];
	REAL * cx;
	for(int i=0;i<(n*d);i++)
	{
		hx[i] = (REAL)px[i];
	}
	cudaMalloc( (void**)&cx, n*d*SIZE_REAL);
	cudaMemcpy(cx, hx, n*d*SIZE_REAL, cudaMemcpyHostToDevice);
	delete [] hx;
	return cx;
};

REAL * cuda_load_param(int k, int d,
		double* pi, double* mu, double* sigma)
{
	// clac number of elements
	int LTS = ((d*(d+1))/2) ; // also used in filling matrix Sigma
	int size = k*d+k*LTS+k+k;// #mu*mu.dim + #sig*sig.size+#p + #logdet
	// band of mu, inv_chol (lower triangular), p, logdet
	REAL * hp = new REAL[size];
	REAL * cp;

	//convert sigmas to inv_chols
	vector<LowerTriangularMatrix> InvChol;
	vector<REAL> logdet;
	SpecialFunctions2 msf;
	;
	for(int m=0; m<k; m++)
	{
		SymmetricMatrix Sigma(d);
		LowerTriangularMatrix L;

		for(int i = 0; i<d;++i)
		{
			for(int j=0; j<=i; ++j)
			{
				int pos = m*d*d+i*d+j;
				Sigma(i+1,j+1) = sigma[pos];
			};
		};

		L = Cholesky(Sigma);

		InvChol.push_back(L.i());
		logdet.push_back( (REAL) msf.logdet(L));

	}

	// load up hp
	REAL* hTmpPtr = hp;
	for(int i=0; i<k;i++)
	{
		for(int j=0;j<d;j++)
		{
			*hTmpPtr++ = (REAL) mu[i*d+j];
			//std::cout << mu[i*D+j] << ' ';

		}
		double* tInvChol = InvChol[i].Store();
		for(int j=0;j<LTS;j++)
		{
			*hTmpPtr++ = (REAL) tInvChol[j];
			//std::cout << tInvChol[j] << ' ';
		}
		*hTmpPtr++ = (REAL) pi[i];
		*hTmpPtr++ = (REAL) logdet[i];
		//std::cout << pi[i] << ' ' << logdet[i] << std::endl;
	}
//	for(int i = 0; i < size; i++)
//	{
//		std::cout << hp[i] << " ";
//	}
//	std::cout << std::endl;
	cudaMalloc( (void**)&cp, size*SIZE_REAL);
	cudaMemcpy(cp, hp, size*SIZE_REAL, cudaMemcpyHostToDevice);
	delete [] hp;
	return cp;
};

#endif
