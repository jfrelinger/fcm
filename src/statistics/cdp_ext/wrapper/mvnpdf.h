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
	 
#endif /*MVNDPF_H_*/
