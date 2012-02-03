//#include "stdafx.h"
#include "MersenneTwister.h"
#include "SpecialFunctions.h"
#include <math.h>
#include <errno.h>
SpecialFunctions::SpecialFunctions(void)
{
}

SpecialFunctions::~SpecialFunctions(void)
{
}

double SpecialFunctions::gammaln(double x)
{
	double d1 = -5.772156649015328605195174e-1;
	double p1[] = {4.945235359296727046734888e0, 2.018112620856775083915565e2,
			2.290838373831346393026739e3, 1.131967205903380828685045e4, 
			2.855724635671635335736389e4, 3.848496228443793359990269e4, 
			2.637748787624195437963534e4, 7.225813979700288197698961e3};
	double q1[] = {6.748212550303777196073036e1, 1.113332393857199323513008e3, 
			7.738757056935398733233834e3, 2.763987074403340708898585e4, 
			5.499310206226157329794414e4, 6.161122180066002127833352e4, 
			3.635127591501940507276287e4, 8.785536302431013170870835e3};
	double d2 = 4.227843350984671393993777e-1;
	double p2[] = {4.974607845568932035012064e0, 5.424138599891070494101986e2, 
           1.550693864978364947665077e4, 1.847932904445632425417223e5, 
           1.088204769468828767498470e6, 3.338152967987029735917223e6, 
           5.106661678927352456275255e6, 3.074109054850539556250927e6};
    double q2[] = {1.830328399370592604055942e2, 7.765049321445005871323047e3, 
           1.331903827966074194402448e5, 1.136705821321969608938755e6, 
           5.267964117437946917577538e6, 1.346701454311101692290052e7, 
           1.782736530353274213975932e7, 9.533095591844353613395747e6};
    double d4 = 1.791759469228055000094023e0;
    double p4[] = {1.474502166059939948905062e4, 2.426813369486704502836312e6, 
           1.214755574045093227939592e8, 2.663432449630976949898078e9, 
           2.940378956634553899906876e10, 1.702665737765398868392998e11, 
           4.926125793377430887588120e11, 5.606251856223951465078242e11};
    double q4[] = {2.690530175870899333379843e3, 6.393885654300092398984238e5, 
           4.135599930241388052042842e7, 1.120872109616147941376570e9, 
           1.488613728678813811542398e10, 1.016803586272438228077304e11, 
           3.417476345507377132798597e11, 4.463158187419713286462081e11};
    double c[] = {-1.910444077728e-03, 8.4171387781295e-04, 
          -5.952379913043012e-04, 7.93650793500350248e-04, 
          -2.777777777777681622553e-03, 8.333333333333333331554247e-02, 
		  5.7083835261e-03};
	if ( (x > 0) && (x <= 2.2204e-016)) {  //x < eps
		return  -log(x);
	} else if ((x > 2.2204e-016) && ( x <= 0.5)) {
		double xden = 1;
		double xnum = 0;  
		for (int i = 0; i < 8; i++) {
			xnum = xnum * x + p1[i];
			xden = xden * x + q1[i];
		}
		return -log(x) + (x * (d1 + x * (xnum / xden)));
	} else if((x > 0.5) && (x <= 0.6796875)) {
		double xm1 = (x - 0.5) - 0.5;
		double xden = 1;
		double xnum = 0;
		for (int i = 0; i < 8; i++) {
			xnum = xnum * xm1 + p2[i];
			xden = xden * xm1 + q2[i];
		}
		return -log(x) + xm1 * (d2 + xm1 * (xnum / xden));
	} else if ((x > 0.6796875) && (x <= 1.5)) {
		double xm1 = (x - 0.5) - 0.5;
		double xden = 1;
		double xnum = 0;
		for (int i = 0; i < 8; i++) {
			xnum = xnum * xm1 + p1[i];
			xden = xden * xm1 + q1[i];
		}
		return xm1 * (d1 + xm1 * (xnum / xden));
	} else if ((x > 1.5) && (x <= 4)) {
		double xm2 = x - 2;
		double xden = 1;
		double xnum = 0;
		for (int i = 0; i < 8; i++) {
			xnum = xnum * xm2 + p2[i];
			xden = xden * xm2 + q2[i];
		}
		return xm2 * (d2 + xm2 * (xnum / xden));
	} else if ((x > 4) && (x <= 12)) {
		double xm4 = x - 4;
		double xden = -1;
		double xnum = 0;
		for (int i = 0; i < 8; i++) {
			xnum = xnum * xm4 + p4[i];
			xden = xden * xm4 + q4[i];
		}
		return d4 + xm4 * (xnum / xden);
	} else {
      double r = c[6];
      double ysq = x * x;
	  for (int i = 0; i < 6; i++) {
		  r = r / ysq + c[i];
	  }
      r = r / x;
      double corr = log(x);
      double spi = 0.9189385332046727417803297;
      return r + spi - 0.5 * corr + x * ( corr - 1);
	}
}

double SpecialFunctions::betaln(double x, double y)
{
	return gammaln(x) + gammaln(y) - gammaln(x + y);
}

/*
 * The standard normal CDF, for one random variable.
 *
 *   Author:  W. J. Cody
 *   URL:   http://www.netlib.org/specfun/erf
 *
 * This is the erfc() routine only, adapted by the
 * transform stdnormal_cdf(u)=(erfc(-u/sqrt(2))/2;
 */
double SpecialFunctions::normcdf(double u) {
	static const double SQRT2 = 1.414213562; 
	static const double a[5] = {
		1.161110663653770e-002,3.951404679838207e-001,2.846603853776254e+001,
		1.887426188426510e+002,3.209377589138469e+003
	};
	static const double b[5] = {
		1.767766952966369e-001,8.344316438579620e+000,1.725514762600375e+002,
		1.813893686502485e+003,8.044716608901563e+003
	};
	static const double c[9] = {
		2.15311535474403846e-8,5.64188496988670089e-1,8.88314979438837594e00,
		6.61191906371416295e01,2.98635138197400131e02,8.81952221241769090e02,
		1.71204761263407058e03,2.05107837782607147e03,1.23033935479799725E03
	};
	static const double d[9] = {
		1.00000000000000000e00,1.57449261107098347e01,1.17693950891312499e02,
		5.37181101862009858e02,1.62138957456669019e03,3.29079923573345963e03,
		4.36261909014324716e03,3.43936767414372164e03,1.23033935480374942e03
	};
	static const double p[6] = {
		1.63153871373020978e-2,3.05326634961232344e-1,3.60344899949804439e-1,
		1.25781726111229246e-1,1.60837851487422766e-2,6.58749161529837803e-4
	};
	static const double q[6] = {
		1.00000000000000000e00,2.56852019228982242e00,1.87295284992346047e00,
		5.27905102951428412e-1,6.05183413124413191e-2,2.33520497626869185e-3
	};
	double y, z;

	y = fabs(u);
    if (y <= 0.46875*SQRT2) {
		/* evaluate erf() for |u| <= sqrt(2)*0.46875 */
		z = y*y;
		y = u*((((a[0]*z+a[1])*z+a[2])*z+a[3])*z+a[4])
			/((((b[0]*z+b[1])*z+b[2])*z+b[3])*z+b[4]);
		return 0.5+y;
	}
	z = exp(-y*y/2)/2;
	if (y <= 4.0) {
		/* evaluate erfc() for sqrt(2)*0.46875 <= |u| <= sqrt(2)*4.0 */
		y = y/SQRT2;
		y = ((((((((c[0]*y+c[1])*y+c[2])*y+c[3])*y+c[4])*y+c[5])*y+c[6])*y+c[7])*y+c[8])
			/((((((((d[0]*y+d[1])*y+d[2])*y+d[3])*y+d[4])*y+d[5])*y+d[6])*y+d[7])*y+d[8]);

		y = z*y;
    } else {
		/* evaluate erfc() for |u| > sqrt(2)*4.0 */
		z = z*SQRT2/y;
		y = 2/(y*y);
        y = y*(((((p[0]*y+p[1])*y+p[2])*y+p[3])*y+p[4])*y+p[5])
			/(((((q[0]*y+q[1])*y+q[2])*y+q[3])*y+q[4])*y+q[5]);
        y = z*(1/sqrt(3.1415926535)-y);
    }
	return (u < 0.0 ? y : 1-y);
}
 //Inverse stdnorm cdf
double SpecialFunctions::norminv(double p)
{
	/* Coefficients in rational approximations. */
	static const double LOW = 0.02425;
	static const double HIGH = 0.97575;
	static const double a[] =
	{
		-3.969683028665376e+01,
		 2.209460984245205e+02,
		-2.759285104469687e+02,
		 1.383577518672690e+02,
		-3.066479806614716e+01,
		 2.506628277459239e+00
	};

	static const double b[] =
	{
		-5.447609879822406e+01,
		 1.615858368580409e+02,
		-1.556989798598866e+02,
		 6.680131188771972e+01,
		-1.328068155288572e+01
	};

	static const double c[] =
	{
		-7.784894002430293e-03,
		-3.223964580411365e-01,
		-2.400758277161838e+00,
		-2.549732539343734e+00,
		 4.374664141464968e+00,
		 2.938163982698783e+00
	};

	static const double d[] =
	{
		7.784695709041462e-03,
		3.224671290700398e-01,
		2.445134137142996e+00,
		3.754408661907416e+00
	};

	double q, r;

	errno = 0;

	if (p < 0 || p > 1)
	{
		errno = EDOM;
		return 0.0;
	}
	else if (p == 0)
	{
		errno = ERANGE;
		return -HUGE_VAL /* minus "infinity" */;
	}
	else if (p == 1)
	{
		errno = ERANGE;
		return HUGE_VAL /* "infinity" */;
	}
	else if (p < LOW)
	{
		/* Rational approximation for lower region */
		q = sqrt(-2*log(p));
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
	}
	else if (p > HIGH)
	{
		/* Rational approximation for upper region */
		q  = sqrt(-2*log(1-p));
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
	}
	else
	{
		/* Rational approximation for central region */
    		q = p - 0.5;
    		r = q*q;
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
	}
}

void SpecialFunctions::cmpower2(int nSize, double *px, double* py, double* pResult) {
	for (int i = 0; i < nSize; i++) {
		pResult[i] = pow(px[i], py[i]);
	}
}

void SpecialFunctions::cmrand(int nSize, MTRand& mt, double* pResult)
{
	for (int i = 0; i < nSize; i++) { 
		pResult[i] =  1- mt.randExc(); //a number in (0,1];
	}
}
bool SpecialFunctions::gammarand(double a, double b, int nSize, MTRand& mt, vector<double>& result)
{
	result.clear();
	//Return false if a or b is not positive.
	if ((a <= 0) || (b <= 0) || (nSize < 1)) return false;

	//If a == 1, then gamma is exponential. (Devroye, page 405).
	if (a == 1) {
		double * pRand = new double[nSize];
		cmrand(nSize,mt,pRand);
		for (int i = 0; i < nSize; i++) {
			result.push_back(-b * log(pRand[i]));
		}
		delete [] pRand;
		pRand = NULL;
	}
	//common working variables
	double *pU = new double[nSize];
	double *pV = new double[nSize];
	double *pX = new double[nSize];
	double *pY = new double[nSize];
	
	//Devroye, page 418 Johnk's generator
	if ((a < 1) && (a > 0)) {
		double c = 1.0 / a;
		double d = 1.0 / (1 - a);

		double *pC = new double[nSize];
		double *pD = new double[nSize];
		int i;
		for (i = 0; i < nSize; i++) {
			pC[i] = c;
			pD[i] = d;
		}
		int nRemain = nSize;

		while (nRemain) {
			cmrand(nRemain,mt, pU);
			cmrand(nRemain,mt, pV);
			cmpower2(nRemain, pU, pC, pX);
			cmpower2(nRemain, pV, pD, pY);
			int nAccept = 0;
			for (i = 0; i < nRemain; i++) {
				if (pX[i] + pY[i] <= 1.0) {
					pX[nAccept] = pX[i];
					pY[nAccept] = pY[i];
					nAccept++;
				}
			}
			if (nAccept) {
				double *pE = new double[nAccept];
				cmrand(nAccept, mt,pE);
				for (i = 0; i < nAccept; i++) {
					pE[i] = -log(pE[i]);
					result.push_back(b * pE[i] * pX[i] / (pX[i] + pY[i]));
					nRemain--;
				}
				delete [] pE;
				pE = NULL;
			}
		}
		delete [] pC;
		pC = NULL;

		delete [] pD;
		pD = NULL;
	}
	
	//Use a rejection method for a > 1
	//Devroye, page 410 Best's algorithm
	if (a > 1) {
		double bb = a - 1;
		double c = 3 * a - 0.75;
		
		int nRemain = nSize;
		double *pZ = new double[nSize];
		double *pW = new double[nSize];
		while (nRemain) {
			cmrand(nRemain, mt,pU);
			cmrand(nRemain, mt,pV);
			int i;
			for (i = 0; i < nRemain; i++) {
				pW[i] = pU[i] * (1 - pU[i]);
				pY[i] = sqrt(c/ pW[i]) * (pU[i] - 0.5);
				pX[i] = bb + pY[i];
				pZ[i] = 64 * pW[i] * pW[i] * pW[i] * pV[i] * pV[i];
			}
			int nAccept = 0;
			for (i = 0; i < nRemain; i++) {
				if (pX[i] >=0) {
					pU[nAccept] = pU[i];
					pV[nAccept] = pV[i];
					pW[nAccept] = pW[i];
					pZ[nAccept] = pZ[i];
					pX[nAccept] = pX[i];
					pY[nAccept] = pY[i];
					nAccept++;
				}
			}
			if (nAccept) {
				for (int i = 0; i < nAccept; i++) {
					if (pZ[i] <= (1.0 - 2.0 * pY[i] * pY[i] / pX[i])) {
						result.push_back(b * pX[i]);
						nRemain--;
					} else {
						if (log(pZ[i])<= 2 * (bb * log(pX[i] / bb) - pY[i])) {
							result.push_back(b * pX[i]);
							nRemain--;
						}
					}
				}
			}
		}
		delete [] pZ;
		pZ = NULL;

		delete [] pW;
		pW = NULL;
	}
	delete [] pX;
	pX = NULL;
	delete [] pY;
	pY = NULL;
	delete [] pU;
	pU = NULL;
	delete [] pV;
	pV = NULL;
	return true;
}

double SpecialFunctions::chi2rand(double a, MTRand& mt) {
	return gammarand(a/2,2,mt);
}
double SpecialFunctions::gammarand(double a, double b, MTRand& mt)
{

	//Return -1 if a or b is not positive.
	if ((a <= 0) || (b <= 0)) return -1.0;

	//If a == 1, then gamma is exponential. (Devroye, page 405).
	if (a == 1) {
		return -b * log(1 - mt.randExc());
	}
	
	//Devroye, page 418 Johnk's generator
	if ((a < 1) && (a > 0)) {
		double c = 1.0 / a;
		double d = 1.0 / (1 - a);
		while (1) {
			double dX = pow(1-mt.randExc(), c);
			double dY = pow(1-mt.randExc(), d);
			if (dX + dY <= 1.0) {
				double dE = -log(1 - mt.randExc());
				return b * dE * dX / (dX + dY );    
			}	
		}
	}
	
	//Use a rejection method for a > 1
	//Devroye, page 410 Best's algorithm
	if (a > 1) {
		double bb = a - 1;
		double c = 3 * a - 0.75;
		
		while (1) {
			double dU = 1-mt.randExc();
			double dV = 1-mt.randExc();
			
			double dW = dU * (1 - dU);
			double dY = sqrt(c / dW) * (dU - 0.5);
			double dX = bb + dY;
			if (dX >=0) {
				double dZ = 64 * dW * dW * dW * dV * dV;
				if (dZ <= (1.0 - 2.0 * dY * dY / dX)) {
					return b * dX;
				} else {
					if (log(dZ)<= 2 * (bb * log(dX / bb) - dY)) {
						return b * dX;
					}
				}
			}
		}
	}
	return -1.0;
}

bool SpecialFunctions::betarand(double a, double b, int nSize, MTRand& mt, vector<double>& result)
{
	result.clear();
	if ((a <= 0) || (b <= 0) || (nSize < 1)) return false;
	vector<double> ra, rb;
	if (!gammarand(a, 1, nSize,mt, ra)) return false;
	if (!gammarand(b, 1, nSize, mt, rb)) return false;
	for (int i = 0; i < nSize; i++) {
		result.push_back(ra[i] / (ra[i] + rb[i]));
	}
	return true;
}
double SpecialFunctions::betarand(double a, double b, MTRand& mt)
{
	if ((a <= 0) || (b <= 0)) return -1.0;
	double ra = gammarand(a, 1, mt);
	double rb = gammarand(b, 1, mt);
	return ra / (ra + rb);
}
double SpecialFunctions::betapdf(double x, double a, double b, int logspace)
{
  double d = 0;
  d = gammaln(a+b) - gammaln(a) - gammaln(b) + (a-1)*log(x)+(b-1)*log(1-x);
  if(!logspace)
    d = exp(d);
  return d;
}
double SpecialFunctions::gammainc(double x, double a)
{
	if (x <= 0 || a <= 0) return 0.0;
    
    double amax = 1048576.0; //2^20;
	if (a > amax)  //Approximation for a > amax.  Accurate to about 5.e-5.
	{
		double onethird = 1.0 / 3.0;
		x = amax - onethird + sqrt(amax/a)*(x-(a-onethird));
		if (x<0) return 0.0; 
		a = amax;
	}
	double eps = 2.2204e-014; //2^(-52) * 100;
	//Series expansion for x < a+ 1
	if (x < a + 1)
	{
		double ap = a;
		double del = 1.0;
		double sum = 1.0;
		while ( del >= eps * sum)
		{
			ap += 1.0;
			del *= x / ap;
			sum += del;
		}
		return sum * exp(-x + a * log(x)-gammaln(a+1));
	} else {
		double a0 = 1.0;
		double a1 = x;
		double b0 = 0.0;
		double b1 = 1.0;
		double fac = 1.0 / a1;
		double n = 1;
		double g = b1 * fac;
		double gold = b0;
		while ( fabs(g-gold) >= eps * fabs(g))
		{
			gold = g;
			double ana = n - a;
			a0 = ( a1 + a0 * ana) * fac;
			b0 = ( b1 + b0 * ana) * fac;
			double anf = n * fac;
			a1 = x * a0 + anf * a1;
			b1 = x * b0 + anf * b1;
			fac = 1 / a1;
			g = b1 * fac;
			n += 1;
		}
		return 1.0 - exp(-x + a * log(x) - gammaln(a)) * g;
	}
}
double SpecialFunctions::gammacdf(double x,double a, double b)
{
	if (x < 0) return 0.0;
	return gammainc(x / b, a);
}
double SpecialFunctions::gammapdf(double x,double a, double b)
{
	double z = x / b;
    double u = (a - 1) * log(z) - z - gammaln(a);
	return (exp(u) / b);
}

double SpecialFunctions::gammainv(double p,double a, double b)
{
	if (p <=0) return 0.0;
	if (a <=0) return 0.0;
	// ==== Newton's Method to find a root of GAMCDF(X,A,B) = P ====
	// Choose a starting guess for q.  Use quantiles from a lognormal
	// distribution with the same mean (==a) and variance (==a) as G(a,1).
	double loga = log(a);
	double sigsq = log(1+a) - loga;
	double mu = loga - 0.5 * sigsq;
	//double q = exp(mu - sqrt(2*sigsq)*erfcinv(2*p)); //original implementation
	double q = exp(mu + sqrt(sigsq)*norminv(p));


	//Limit the number of iterations.
	int maxiter = 500;
	double reltol = 1.8190e-012; //eps(class(q)).^(3/4);
	double F = gammacdf(q,a,1.0);
	double dF = F - p;
	for (int iter = 0; iter < maxiter; iter++) { 
		// Compute the Newton step, but limit its size to prevent stepping to
		// negative or infinite values.
		double f = gammapdf(q,a,1);
		double h = dF / f;
		
		//Avoid taking too large of a step
		double qnew = 10 * q;
		if (q-h < qnew) {qnew = q-h;}
		if (q / 10 > qnew) { qnew = q / 10;}
	    
		// Break out of the iteration loop when the relative size of the last step
		// is small for all elements of q.
		if (fabs(h) <= reltol*q) {
			q = qnew;
			break;
		} else {
			// Check the error, and backstep for those elements whose error did not
			// decrease.  The direction of h is always correct, because f > 0
			double dFold = dF;
			for (int j = 0; j < 25; j++) {
				F = gammacdf(qnew,a,1.0);
				dF = F - p;
				if (fabs(dF) <= fabs(dFold)) {
					break;
				} else {
					qnew = (q + qnew)/2;	
				}
			}
		}
		q = qnew;
	}

	// Add in the scale factor
	return q * b;
}
double SpecialFunctions::binorand(int n, double p, MTRand& mt)
{	
	//generates values using the definition of the binomial
	//distribution, as a sum of Bernoulli random variables.  See Devroye,
	//Lemma 4.1 on page 428, method on page 524.
	double r = 0;
	for (int i=0; i < n; i++) {
		if (mt() < p) r++;
	}
	return r;
}


