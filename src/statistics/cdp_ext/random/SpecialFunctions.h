#pragma once
#include <vector>
class SpecialFunctions
{
public:
	SpecialFunctions(void);
	~SpecialFunctions(void);

	static double gammaln(double x);
	static double betaln(double x, double y);

	static double norminv(double p);			//inverse normal cdf
	static double normcdf(double u);

	static double gammainc(double x, double a);
	static double gammacdf(double x,double a, double b);
	static double gammainv(double p,double a, double b);
	static double gammapdf(double x,double a, double b);
	
	static void cmpower2(int nSize, double *px, double* py, double* pResult);
	static void cmrand(int nSize, MTRand& mt, double* pResult);
	static bool gammarand(double a, double b, int nSize, MTRand& mt, vector<double>& result);
	static double gammarand(double a, double b, MTRand& mt);
	static double chi2rand(double a, MTRand& mt);
	static bool betarand(double a, double b, int nSize, MTRand& mt, vector<double>& result);
	static double betarand(double a, double b, MTRand& mt);
	double betapdf(double x, double a, double b,int logspace);
	static double binorand(int n, double p, MTRand& mt);
};
