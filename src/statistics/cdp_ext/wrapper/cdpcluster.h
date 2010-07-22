#ifndef CDPCLUSTER_H_
#define CDPCLUSTER_H_

#include "MersenneTwister.h"
#include "Model.h"
#include "cdp.h"

//int main(int argc,char* argv[]);

class cdpcluster{
	public:
	virtual ~cdpcluster(void); 
	cdpcluster(int n, int d, double* x);
	void makeResult();
	void run();
	void step();
	void stepburn();
	void setseed(int x);
	void setT(int t);
	void setJ(int j);
	void setBurnin(int t);
    void setIter(int t);
    void setVerbose(bool verbosity);
	int getBurnin();
    int getIter();
	int getn();
	int getd();
	int getclustN();
	int getT();
	int getJ();
	int getK(int idx);
	void getK(int d, int* res);
	void setlambda0( double lambda0 );
	double getlambda0();
	void setphi0( double phi0 );
	double getphi0();
	void setnu0( double nu0 );
	double getnu0();
	void setgamma( double gamma );
	double getgamma();
	void setnu( double nu );
	double getnu();
	void sete0( double e0 );
	double gete0();
	void setf0( double f0 );
	double getf0();
	void setee( double ee );
	double getee();
	void setaa( double aa );
	double getaa();
	void setff( double ff );
	double getff();	
	double getMu(int idx, int pos);
	double getm(int idx, int pos);
	double getSigma(int i, int j, int k);
	double getPhi(int i, int j , int k);
	double getp(int idx);
	void setgpunchunksize(int x);
	int getgpunchunksize();
	void setdevice(int x);
	int getdevice();
	int getnumberdevices();
	void setnumberdevices(int x);
	
	bool samplem();
	void samplem(bool x);
	bool samplePhi();
	void samplePhi(bool x);
	bool samplew();
	void samplew(bool x);
	bool sampleq();
	void sampleq(bool x);
	bool samplealpha0();
	void samplealpha0(bool x);
	bool samplemu();
	void samplemu(bool x);
	bool sampleSigma();
	void sampleSigma(bool x);
	bool samplek();
	void samplek(bool x);
	bool samplep();
	void samplep(bool x);
	bool samplealpha();
	void samplealpha(bool x);

	void loadalpha0(double x);
	void loadMu(int n, int d, double* x);
	void loadm(int n, int d, double* x);
	void loadp(int n, int d, double* x);
	void loadpV(int n, int d, double* x);
	void loadSigma(int i, int j, int k , double* x);
	void loadPhi(int i, int j, int k , double* x);
	void loadW(int i, double* x);
	void loadK(int i, double* x);
	void loadq(int i, double* x);
	void loadqV(int i, double* x);
	void loadalpha(int i, double* x);
	
	void printModel();



	private:
	void loadRows(double* from, int* to, int cols);
	void loadRowsCols(double* from, vector<RowVector>& to, int rows, int cols);
	void loadRowsCols(double* from, vector<SymmetricMatrix>& to, int idx, int rows, int columns);
	void loadRows(double* from, RowVector& A, int cols);
	bool resultInit;
	CDPResult* param;
	//CDPResult result;
	Model model;
	CDP cdp;
	MTRand mt;
	bool verbose;
};

#endif /*CDPCLUSTER_H_*/
