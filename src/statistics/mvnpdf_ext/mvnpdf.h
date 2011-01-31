#ifndef MVNPDF_H_
#define MVNPDF_H_

#include "MersenneTwister.h"
#include "newmat.h"
#include "specialfunctions2.h"


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
void load_data(int n, int d, int pad, double *x, float *out);
void pack_param(int d, int k, double* mu, double* sigma, float *out);
int next_mult(int k, int p);
int pack_size(int n, int d);
int pad_data_dim(int n, int d);
void load_data(int n, int d, int pad, double *x, double *out);


#endif /*cuda*/
#endif /*MVNDPF_H_*/
