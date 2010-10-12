#ifndef MVNPDF_H_
#define MVNPDF_H_

#include "MersenneTwister.h"
#include "newmat.h"
#include "specialfunctions2.h"

#if defined(CDP_CUDA)
#include "CDPBaseCUDA.h"
#endif



double mvnpdf(int xd, double* px,
	 int md, double* mu,
	 int sd, int sp, double* sigma);
	 
void mvnpdf(int xd, int xp, double* px, 
	 int md, double* mu,
	 int sd, int sp, double* sigma,
	 int outd, double* out);
	 
double wmvnpdf(int xd, double* px,
	 double pi,
	 int md, double* mu,
	 int sd, int sp, double* sigma);
	 
void wmvnpdf(int xd, int xp, double* px, 
	 double pi,
	 int md, double* mu,
	 int sd, int sp, double* sigma,
	 int outd, double* out);
	 
void wmvnpdf(int xd, int xp, double* px, 
	int pd, double* pi,
	int md, int mp, double*mu,
	int sk, int sd, int sp, double* sigma,
	int outd, double* out);

#if defined(CDP_CUDA)
void cuda_wmvnpdf(int n, int d, int k,
		double* px, double* pi, double* mu, double* sigma,
		double* out);
REAL * cuda_load_data(int n, int d, double* px);
REAL * cuda_load_param(int k, int d,
				double* pi, double* mu, double* sigma);
#endif
#endif /*MVNDPF_H_*/
