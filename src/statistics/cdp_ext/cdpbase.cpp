#include "stdafx.h"
#include "MersenneTwister.h"
#include "specialfunctions2.h"
#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "cdpprior.h"
#include "cdpbase.h"

CDPBase::CDPBase()
{
}

CDPBase::~CDPBase(void)
{
}
double CDPBase::sampleAlpha(double* V, double e, double f, MTRand& mt) {
	int firstone;
	double newf;
	newf = f;
	for(firstone=0;V[firstone]<1;firstone++) {
		newf-=log(1-V[firstone]);
	}
	return msf.gammarand(e+firstone,1.0/newf,mt);
}

#if defined(CDP_TBB)
// QUANLI -- need to fix this for the parallel version, I'm not sure if I've changed it correctly
  int CDPBase::sampleW(RowVector& x, RowVector& q, concurrent_vector<RowVector>&p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, concurrent_vector<double>& logdet, MTRand& mt)
#else
  int CDPBase::sampleW(RowVector& x, RowVector& q, vector<RowVector>& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, vector<double>& logdet, MTRand& mt)
#endif
  {
    double* weights = new double[q.Ncols()];
    int J = q.Ncols();
    int T = p[0].Ncols();
    int D = x.Ncols();
    for(int j=0;j<J;j++)
      {
	weights[j]=0;
	for(int t=0;t<T;t++)
	  {
	    Real* px = x.Store();
	    Real* pmu = mu[j*T+t].Store();
	    double d = msf.mvnormpdf(px,pmu,Sigma[t+j*T],D,1,logdet[t+j*T]);
	    weights[j]+=exp(log(p[j][t])+d);
	  }
	
	weights[j] = exp(log(q[j])+log(weights[j]));
      }
    
    int r= sample(weights,q.Ncols(),mt);
    delete [] weights;
    return r;
  }


#if defined(CDP_TBB)
int CDPBase::sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, concurrent_vector<double>& logdet, MTRand& mt) 
#else
  int CDPBase::sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, vector<double>& logdet, MTRand& mt) 
#endif
{
  int T = p.Ncols();
  int D = x.Ncols();
  int i; 
  double* weights = new double[T];
  
  for (i=0; i < T; i++) {
    Real* px = x.Store();
    Real* pmu = mu[i].Store();
    double d = msf.mvnormpdf(px,pmu,Sigma[i],D,1,logdet[i]);
    weights[i] = exp(log(p[i]) + d);
  }
  int r =  sample(weights,T,mt);
  delete [] weights;
  return r;
}


#if defined(CDP_TBB)
int CDPBase::sampleK(RowVector& x, RowVector& p, concurrent_vector<RowVector>& mu, concurrent_vector<UpperTriangularMatrix>& Sigma, int index,concurrent_vector<double>& logdet, MTRand& mt) 
#else
  int CDPBase::sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<UpperTriangularMatrix>& Sigma, int index,vector<double>& logdet, MTRand& mt) 
#endif
{
  int T = p.Ncols();
  int D = x.Ncols();
  int i; 
  double* weights = new double[T];
  
  for (i=0; i < T; i++) {
    Real* px = x.Store();
    Real* pmu = mu[i+index * T].Store();
    double d = msf.mvnormpdf(px,pmu,Sigma[i+index * T],D,1,logdet[i+index * T]);
    weights[i] = exp(log(p[i]) + d);
  }
  int r =  sample(weights,T,mt);
  delete [] weights;
  return r;
}

int CDPBase::sample(double* w, int n, MTRand& mt) {
  int i;
  double dsum = 0;
  for (i = 0; i < n;i++) {
    dsum+=w[i];
  }
  w[0] /=dsum;
  for (i = 1; i < n;i++) {
    w[i] =w[i] / dsum + w[i-1];
  }
  double d = mt();
  int k;
  for(k=0;d>w[k];k++)
    ;
  //there are rare cases k >=n due to round up errors
  if (k >n-1) { k = n-1; }
  return k;				//zero based index
}


void CDPBase::sampleMuSigma(Matrix& x, int n, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu, SymmetricMatrix& PostSigma,UpperTriangularMatrix& uti, double& logdet, MTRand& mt) {
  int D = m.Ncols();
  SymmetricMatrix mycov;
  double postnu;
  SymmetricMatrix postS;
  RowVector postm;
  double postgamma;
  if (n>0) {
    RowVector xbar(D);
    int nxrow = x.Nrows();
    if (n>1) {
      for (int i = 0; i < D; i++) {
	xbar[i] = x.Column(i+1).Sum() /  nxrow;
      }
      cov(x,xbar,1,mycov);
    } else {
      xbar = x;
      mycov = SymmetricMatrix(D); mycov = 0;
    }
    postnu = nu + 2 + nxrow;
    postS << mycov + Phi * nu + (xbar-m).t() * (xbar-m)*((double)nxrow) / (nxrow * gamma +1) ;
    postm = (m+xbar *nxrow * gamma) / (nxrow * gamma +1);
    postgamma = gamma / (nxrow * gamma +1);
  } else  {
    postnu = nu+2;
    postS << Phi*nu;
    postm = m;
    postgamma = gamma;
  }
  
  SymmetricMatrix mySinv = postS.i();
  LowerTriangularMatrix mySinvL = Cholesky(mySinv);
  PostSigma = msf.invwishartrand((int)postnu,mySinvL,mt);
  
  LowerTriangularMatrix mycovinvL = Cholesky(PostSigma);
  logdet = msf.logdet(mycovinvL);
  uti = mycovinvL.t().i();
  
  mycovinvL << mycovinvL * sqrt(postgamma);
  PostMu = msf.mvnormrand(postm, mycovinvL,mt);
  
}

void CDPBase::samplePhi(int n, double nu, vector<SymmetricMatrix>& Sigma, double nu0, SymmetricMatrix& Lambda0, SymmetricMatrix& newphi, MTRand& mt)
{
  int D = Lambda0.Ncols();
  if(n==0) {
    SymmetricMatrix Lambda(D); Lambda << Lambda0/ nu0;
    //std::cout << Lambda << endl;
    LowerTriangularMatrix L = Cholesky(Lambda);
    newphi = msf.wishartrand((int)nu0,L,mt);
  } else {
    double nuStar = nu0 + n * (nu +D + 1);
    int i;
    SymmetricMatrix postcov(D); postcov << Lambda0.i() * nu0; 
    for (i = 0; i < n; i++) {
      postcov += Sigma[i].i() * nu;
    }
    SymmetricMatrix LambdaStar(D); LambdaStar << postcov.i();
    LowerTriangularMatrix L = Cholesky(LambdaStar);
    newphi = msf.wishartrand((int)nuStar,L,mt);
  }
}

void CDPBase::sampleMuSigma(vector<int>& indexes, double nu, double gamma,RowVector& m, SymmetricMatrix& Phi, RowVector& PostMu, SymmetricMatrix& PostSigma,UpperTriangularMatrix& uti, double& logdet, MTRand& mt) {
  Matrix x;
  int j;
  int n = (int)indexes.size();
  if (n> 0) {
    x = Matrix(n,prior.D);
    int count = 0;
    vector<int>::iterator it;
    for(it=indexes.begin(); it != indexes.end(); ++it) {
      for (j = 0; j < prior.D; j++) {
	x[count][j] = mX[*it][j];
      }
      count++;
    }
  }
  sampleMuSigma(x,n,nu,gamma,m,Phi,PostMu,PostSigma,uti,logdet,mt);
}

void CDPBase::cov(Matrix& x, RowVector& mu,int mul, SymmetricMatrix& result) {
	int D = mu.Ncols();
	int nrows = x.Nrows();
	result = SymmetricMatrix(D);result = 0;
	int i;
	for (i = 1; i <=D; i++) {
		x.Column(i) = x.Column(i) - mu[i-1];
	}
	result << x.t() * x;
	if (mul==0) {
		result = result / (nrows-1);
	}
}

void CDPBase::sampleP(vector<int>& p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt) {
  double* a = new double[T];
  double* b = new double[T];
  int i;
  for (i = 0; i< T; i++) {
    a[i] = 0;
    b[i] = 0;
  }
  if(n>0) {
    for (i = 0; i < n;i++) {
      a[(int)p[i]]++;
    }
    b[T-1] = 0;
    double sum = 0;
    for (i = T-2; i >=0;i--) {
      b[i] = b[i+1] + a[i+1];
    }       
  } 
  postV = RowVector(T);postV = 0;
  for (i = 0; i< T; i++) {
    postV[i] = msf.betarand(1+a[i],gamma + b[i],mt);
    if (postV[i]==1) {
      break;
    } 
  }
  if (postV[T-1] > 0) {
    postV[T-1] = 1.0;
  }
  postp = RowVector(T);postp[0] = postV[0];   
  double prod = 1;
  for(i = 1; i < T; i++) {
    prod *= (1-postV[i-1]);
    postp[i] = prod*postV[i];
  }
}

void CDPBase::sampleP(int* p, int n, double gamma, int T, RowVector& postp, RowVector& postV, MTRand& mt) {
  double* a = new double[T];
  double* b = new double[T];
  int i;
  for (i = 0; i< T; i++) {
    a[i] = 0;
    b[i] = 0;
  }
  if(n>0) {
    for (i = 0; i < n;i++) {
      a[p[i]]++;
    }
    b[T-1] = 0;
    double sum = 0;
    for (i = T-2; i >=0;i--) {
      b[i] = b[i+1] + a[i+1];
    }       
  } 
  postV = RowVector(T);postV = 0;
  for (i = 0; i< T; i++) {
    postV[i] = msf.betarand(1+a[i],gamma + b[i],mt);
    if (postV[i]==1) {
      break;
    } 
  }
  if (postV[T-1] > 0) {
    postV[T-1] = 1.0;
  }
  postp = RowVector(T);postp[0] = postV[0];   
  double prod = 1;
  for(i = 1; i < T; i++) {
    prod *= (1-postV[i-1]);
    postp[i] = prod*postV[i];
  }
}

void CDPBase::sampleM(int n, double gamma, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma, RowVector& m0, SymmetricMatrix& Phi0, RowVector& newm, MTRand& mt) {
  int D = Phi0.Ncols();
  newm = RowVector(D);
  int i;
  if(n==0) {
    LowerTriangularMatrix L = Cholesky(Phi0);
    newm = msf.mvnormrand(m0, L, mt);
  } else {
    SymmetricMatrix SigmaStarInv(D);SigmaStarInv = 0;
    RowVector muStar(D); muStar = 0;
    for (i = 0; i < n; i++) {
      SymmetricMatrix s(D); 
      s = Sigma[i] * gamma;
      s= s.i();
      SigmaStarInv += s;
      muStar += mu[i] * s;
    }
    SymmetricMatrix SigmaStar(D); SigmaStar = SigmaStarInv.i();
    muStar = muStar * SigmaStar;
    SymmetricMatrix Phi0Inv(D);Phi0Inv = Phi0.i();
    SymmetricMatrix Phi0Star(D);  Phi0Star = (Phi0Inv + SigmaStarInv).i();
    RowVector m0Star(D); m0Star = (m0 * Phi0Inv + muStar * SigmaStarInv) * Phi0Star;
    LowerTriangularMatrix L = Cholesky(Phi0Star);
    newm = msf.mvnormrand(m0Star, L, mt);
  }
}


/*
double CDPBase::sampleWj(RowVector& x, double q, RowVector& m, SymmetricMatrix& Phi, double nu, double gamma) {
	
	SymmetricMatrix tmpcov;
	tmpcov << Phi*nu*(gamma+1)/(nu+2.0);
	try {
		LowerTriangularMatrix LSigma = Cholesky(tmpcov);
		Real* ps = x.Store();
		Real* pm = m.Store();
		double d = msf.mvtpdf(ps,pm,LSigma,nu+2,Phi.Ncols(),1);
		return exp(d + log(q));
	} catch (NPDException) {
		std::cout << "Singular Covariance Matrix in Cholesky Decomposition. Program aborted" << endl;
		exit(1);
	}	
}
*/
/*
int CDPBase::sampleW(RowVector& x, RowVector& q, vector<RowVector>& m, vector<SymmetricMatrix>& Phi, double nu, double gamma) {
	int i;
	double* weights = new double[m.size()];
	for (i = 0; i < (int)m.size();i++) {
		weights[i] = sampleWj(x,q[i],m[i],Phi[i],nu,gamma);
	}
	int r= sample(weights,(int)m.size());
	delete [] weights;
	return r;
}
*/
/*

int CDPBase::sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma) {
	int T = p.Ncols();
	int D = x.Ncols();
	int i; 
	double* weights = new double[T];

	for (i=0; i < T; i++) {
		LowerTriangularMatrix LSigma = Cholesky(Sigma[i]);
		Real* px = x.Store();
		Real* pmu = mu[i].Store();
		double d = msf.mvnormpdf(px,pmu,LSigma,D,1);
		weights[i] = exp(log(p[i]) + d);
	}
	int r =  sample(weights,T);
	delete [] weights;
	return r;
}
int CDPBase::sampleK(RowVector& x, RowVector& p, vector<RowVector>& mu, vector<SymmetricMatrix>& Sigma, int index) {
	int T = p.Ncols();
	int D = x.Ncols();
	int i; 
	double* weights = new double[T];

	for (i=0; i < T; i++) {
		LowerTriangularMatrix LSigma = Cholesky(Sigma[i+index * T]);
		Real* px = x.Store();
		Real* pmu = mu[i+index * T].Store();
		double d = msf.mvnormpdf(px,pmu,LSigma,D,1);
		weights[i] = exp(log(p[i]) + d);
	}
	int r =  sample(weights,T);
	//std::cout << r << endl;
	delete [] weights;
	return r;
}
*/
