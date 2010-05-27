#ifndef MVNPDF_H_
#define MVNPDF_H_

#include "MersenneTwister.h"
#include "newmat.h"
#include "specialfunctions2.h"



double mvnpdf(int xd, double* px,
	 int md, double* mu,
	 int sd, int sp, double* sigma);
	 
#endif /*MVNDPF_H_*/
