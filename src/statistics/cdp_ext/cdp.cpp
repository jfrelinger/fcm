#include <math.h>
#include "stdafx.h"
#include "Model.h"
#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "MersenneTwister.h"
#include "specialfunctions2.h"
#include "cdpprior.h"
#include "cdpbase.h"
#include "cdpresult.h"
#include "cdp.h"

#if defined(CDP_TBB)
	#include "cdpt.h"
#endif
CDP::CDP()
{
  // turn off/off individual sampling steps
  mcsamplem = true;
  mcsamplePhi= true;
  mcsamplew= true;
  mcsampleq= true;
  mcsamplealpha0= true;
  
  mcsamplemu= true;
  mcsampleSigma= true;
  mcsamplek= true;
  mcsamplep= true;
  mcsamplealpha= true;
  
}

CDP::~CDP(void)
{
	for (int i = 0; i < prior.N; i++) {
		delete [] mX[i];
	}
	delete [] mX;
}

void CDP::InitMCMCSteps(Model& model)
  {
    mcsamplem = model.samplem;
    mcsamplePhi = model.samplePhi;
    mcsamplew = model.samplew;
    mcsampleq = model.sampleq;
    mcsamplealpha0 = model.samplealpha0;
    mcsamplemu = model.samplemu;
    mcsampleSigma = model.sampleSigma;
    mcsamplek = model.samplek;
    mcsamplep = model.samplep;
    mcsamplealpha = model.samplealpha;
  }

// creates reverse lookup vectors wk2d: observations associated with components 1..JT 
// and w1d: observations associated with clusters 1..J
void CDP::partition(int* W, vector<vector<int> >& w1d)
{
  w1d.clear();
  int i;
  int nT = prior.T;
  int nJ = prior.J;
  vector<int> holder;
  holder.reserve(prior.N);
  w1d.reserve(nJ);
  for (i = 0; i < nJ; i++) {
    w1d.push_back(holder);
  }
  for (i = 0; i < prior.N; i++) {
    w1d[W[i]].push_back(i);
  }
}
void CDP::partition(int* K, vector<int>& w1d, vector<vector<int> >& wk2d)
{
  wk2d.clear();
  int i;
  int nT = prior.T;

  wk2d.reserve(nT);
  vector<int> holder;
  holder.reserve(w1d.size());

  for (i = 0; i < nT; i++) {
    wk2d.push_back(holder);
  }
  
  vector<int>::iterator it;
  
  for (it = w1d.begin(); it!=w1d.end(); it++) 
    {
      wk2d[K[*it]].push_back(*it);
    }
}

// creates the objects KJ and KJunique that store the labels of the nonempty components within 
// the cluster j.  KJ has duplicities and KJUnique does not.  In other words
// KJ[j] = 1 1 1 1 3 3 3 3 (the component membership for observations in cluster j)
// KJUnique[j] = 1 3 (the nonempty components in cluster j)
// only list the component membership for a single cluster characterized by w1d
void CDP::UpdateKJ(int* K,vector<int>& w1d, vector<int>& KJ) {
  KJ.clear();
  if ((int) w1d.size()> 0) 
    {
      vector<int>::iterator it;
      for(it=w1d.begin(); it != w1d.end(); ++it) 
	{
	  KJ.push_back(K[*it]);
	}
    } 
}


bool CDP::SimulateFromPrior(CDPResult& result, MTRand& mt) {
  int i,j,t;
  LowerTriangularMatrix temp;
  //J
  result.alpha0 = msf.gammarand(prior.e0,1.0/prior.f0,mt);

  LowerTriangularMatrix ltcov = Cholesky(prior.Phi0);
  SymmetricMatrix lambda = prior.Lambda0 / prior.nu0;
  LowerTriangularMatrix ltlambda = Cholesky(lambda);
  for (j = 0; j < prior.J; j++) {
    // m_j ~ N(m0,Phi0)
    result.m.push_back(msf.mvnormrand(prior.m0,ltcov,mt));	
    // Phi_j ~ W(nu0,Lambda0/nu0)
    result.Phi.push_back(msf.wishartrand((int)prior.nu0,ltlambda,mt));
    
    temp = Cholesky(result.Phi[j]);
    temp << temp * sqrt(prior.nu*(prior.gamma+1)/(prior.nu+2.0));
    
    // log_determinant of Chol(Phi_j*nu*(gamma+1)/(nu+2))
    result.Phi_log_det.push_back(msf.logdet(temp));
    
    // inverse of transpose of Chol(Phi_j*nu*(gamma+1)/(nu+2))
    result.Phi_T_i.push_back(temp.t().i());
    
    // alpha_j ~ Ga(ee,ff)
    result.alpha[j] = msf.gammarand(prior.ee,1.0/prior.ff,mt);
  }
  
  //J by T
  int dummy;
  RowVector p;
  RowVector pV;
  for (j = 0; j < prior.J;j++) {
    SymmetricMatrix phi = result.Phi[j] * prior.nu;
    LowerTriangularMatrix ltphi = Cholesky(phi);
    for (t = 0; t < prior.T;t++) {
      int index = result.GetIndex(j,t);
      
      // Sigma_j,t ~ IW(nu+2,nu*Phi[j])
      result.Sigma.push_back(msf.invwishartrand((int)(prior.nu + 2),ltphi,mt));
      
      temp << Cholesky(result.Sigma[index]);
      // log of determinant of Chol(Sigma_j,t)
      result.Sigma_log_det.push_back(msf.logdet(temp));
      // inverse of transpose of Chol(Sigma_j,t)
      result.Sigma_T_i.push_back(temp.t().i());
      
      SymmetricMatrix temp = result.Sigma[index] *prior.gamma;

      // mu_j,t ~ N(m_j,gamma*Sigma_j,t)
      result.mu.push_back(msf.mvnormrand(result.m[j],temp,mt));
    }
    sampleP(&dummy,0,result.alpha[j],prior.T,p,pV,mt);
    result.p.push_back(p);
    result.pV.push_back(pV);
  }
  
  //J by 1
  sampleP(&dummy,0,result.alpha0,prior.J,p,pV,mt);
  result.q = p;
  result.qV = pV;
  
  for (i = 0; i < prior.N;i++) {
    RowVector row(prior.D);
    for (j =0; j < prior.D; j++) {
      row[j] = mX[i][j];
    }

    result.W[i] = sampleW(row,result.q, result.p, result.mu, result.Sigma_T_i, result.Sigma_log_det, mt);
    int index = result.W[i];
    result.K[i] = sampleK(row,result.p[index],result.mu,result.Sigma_T_i,index,result.Sigma_log_det,mt);
  }
  return true;
}


bool CDP::SimulateFromPrior2(CDPResult& result, MTRand& mt) {
  int i,j,t;
  LowerTriangularMatrix temp;
  //J
  result.alpha0 = 1;//msf.gammarand(prior.e0,1.0/prior.f0,mt);

  LowerTriangularMatrix ltcov = Cholesky(prior.Phi0);
  SymmetricMatrix lambda = prior.Lambda0 / prior.nu0;
  LowerTriangularMatrix ltlambda = Cholesky(lambda);
  for (j = 0; j < prior.J; j++) {
    // m_j ~ N(m0,Phi0)
    result.m.push_back(msf.mvnormrand(prior.m0,ltcov,mt));	
    // Phi_j ~ W(nu0,Lambda0/nu0)
    //    result.Phi.push_back(msf.wishartrand((int)prior.nu0,ltlambda,mt));
    result.Phi.push_back(prior.Lambda0);
    temp = Cholesky(result.Phi[j]);
    temp << temp * sqrt(prior.nu*(prior.gamma+1)/(prior.nu+2.0));
    
    // log_determinant of Chol(Phi_j*nu*(gamma+1)/(nu+2))
    result.Phi_log_det.push_back(msf.logdet(temp));
    
    // inverse of transpose of Chol(Phi_j*nu*(gamma+1)/(nu+2))
    result.Phi_T_i.push_back(temp.t().i());
    
    // alpha_j ~ Ga(ee,ff)
    result.alpha[j] = 1;//msf.gammarand(prior.ee,1.0/prior.ff,mt);
  }
  
  //J by T
  for (j = 0; j < prior.J;j++) {
    SymmetricMatrix phi = result.Phi[j] * prior.nu;
    LowerTriangularMatrix ltphi = Cholesky(phi);
    for (t = 0; t < prior.T;t++) {
      int index = result.GetIndex(j,t);
      
      // Sigma_j,t ~ IW(nu+2,nu*Phi[j])
      //      result.Sigma.push_back(msf.invwishartrand((int)(prior.nu + 2),ltphi,mt));
      result.Sigma.push_back(prior.Lambda0);
      
      temp << Cholesky(result.Sigma[index]);
      // log of determinant of Chol(Sigma_j,t)
      result.Sigma_log_det.push_back(msf.logdet(temp));
      // inverse of transpose of Chol(Sigma_j,t)
      result.Sigma_T_i.push_back(temp.t().i());
      
      SymmetricMatrix temp = result.Sigma[index] *prior.gamma;

      // mu_j,t ~ N(m_j,gamma*Sigma_j,t)
      result.mu.push_back(msf.mvnormrand(result.m[j],temp,mt));
    }
    RowVector p(prior.T);p=1./(double)prior.T;
    RowVector pV(prior.T);pV=0;
    // int dummy;
    //    sampleP(&dummy,0,result.alpha[j],prior.T,p,pV,mt);
    result.p.push_back(p);
    result.pV.push_back(pV);
  }
  
  //J by 1
  //  sampleP(&dummy,0,result.alpha0,prior.J,p,pV,mt);
  result.q = 1./(double) prior.J;
  result.qV = 0;
  
  for (i = 0; i < prior.N;i++) {
    RowVector row(prior.D);
    for (j =0; j < prior.D; j++) {
      row[j] = mX[i][j];
    }

    result.W[i] = sampleW(row,result.q, result.p, result.mu, result.Sigma_T_i, result.Sigma_log_det, mt);
    int index = result.W[i];
    result.K[i] = sampleK(row,result.p[index],result.mu,result.Sigma_T_i,index,result.Sigma_log_det,mt);
  }
  return true;
}


void CDP::CheckSpecialCases(Model& model,CDPResult& result)
{
  // if this is a single cluster model:
  if(prior.J==1)
    {
      mcsamplew=false;
      mcsamplem=false;
      mcsamplePhi=false;
      mcsamplealpha0=false;
      mcsampleq=false;
      
      result.m[0] = prior.m0;
      result.Phi[0] = prior.Lambda0;
      result.q[0] = 1.;
      for(int i=0;i<result.N;i++)
	result.W[i]=0;
      result.alpha0=0;
    }

}

// sampling steps for a single layer DP mixture:
bool CDP::clusterIterate(int clustnum,vector<int>& w1d,CDPResult& result, MTRand& mt)
{
  vector<vector<int> >wk2d;
  partition(result.K, w1d,wk2d); //wk2d is a vector of lists of which observations are associated with which component
  vector<int> KJ;
  UpdateKJ(result.K,w1d,KJ); // KJ is a list of component numbers for each observation in this cluster
  
  //sample mu and sigma
  //J by T
  if(mcsamplemu || mcsampleSigma) {
    //disable second thread layer
    //#if defined(CDP_TBB)
    //	parallel_for(blocked_range<size_t>(0, prior.T),
    //	MuSigmaSampler(this, &result,&clustnum,&wk2d),auto_partitioner());			
    //#else
    int t;
    RowVector PostMu;
    SymmetricMatrix PostSigma;
    UpperTriangularMatrix uti;
    double logdet;
    for (t = 0; t < prior.T;t++) {
      int index = result.GetIndex(clustnum,t);
      
      int flag = 0;
      int ITERTRY = 10;
      do {
	try {
	  sampleMuSigma(wk2d[t],prior.nu,prior.gamma,result.m[clustnum],result.Phi[clustnum],
			PostMu,PostSigma,uti,logdet,mt);
	  flag = 0;
	} catch (NPDException) {
	  flag++;
	  if (flag >= ITERTRY) {
	    std::cout << "SampleMuSigma failed due to singular matrix after 10 tries" << endl;
	    exit(1);
	  }
	}
      } while (flag >0);
      
      if(mcsamplemu) {      
	result.mu[index] = PostMu;
      }
      if(mcsampleSigma) {
	result.Sigma[index] = PostSigma;
	result.Sigma_log_det[index] = logdet;
	result.Sigma_T_i[index] = uti;
      }
    }
    //#endif
  }
  
  //sample p
  RowVector p;
  RowVector pV;
  if(mcsamplep)
    {      
      int n = (int)w1d.size(); 
      sampleP(KJ,n,result.alpha[clustnum],prior.T,p,pV,mt);
      result.p[clustnum] = p;
      result.pV[clustnum] = pV;
    }
  
  
  //sample m, Phi, alpha
  
  vector<RowVector> muj;
  vector<SymmetricMatrix> Sigmaj;
  int n = (int) KJ.size(); // number of observations in cluster j
  //std::cout << n << endl;
  if (n > 0) {
    vector<int>::iterator it;
    for(it=KJ.begin(); it != KJ.end(); ++it) {
      muj.push_back(result.mu[result.GetIndex(clustnum,*it)]); // the occupied component means
      Sigmaj.push_back(result.Sigma[result.GetIndex(clustnum,*it)]); // the occupied component covariances
    }
  }
  
  if(mcsamplem)
    {
      RowVector newmu;
      int flag = 0;
      int ITERTRY = 10;
      do {
	try {
	  sampleM(n,prior.gamma,muj,Sigmaj,prior.m0,prior.Phi0,newmu,mt);
	  flag = 0;
	} catch (NPDException) {
	  flag++;
	  if (flag >= ITERTRY) {
	    std::cout << "SampleM failed due to singular matrix after 10 tries" << endl;
	    exit(1);
	  }
	}
      } while (flag >0);
      result.m[clustnum] = newmu;
    }
  
  if(mcsamplePhi) {
    int flag = 0;
    int ITERTRY = 10;
    do {
      try {
	LowerTriangularMatrix temp;
	SymmetricMatrix newphi;
	samplePhi(n,prior.nu,Sigmaj,prior.nu0,prior.Lambda0,newphi,mt);
	result.Phi[clustnum] = newphi;
	temp = Cholesky(newphi);
	temp << temp * sqrt(prior.nu*(prior.gamma+1)/(prior.nu+2.0));
	result.Phi_log_det[clustnum] = msf.logdet(temp);
	result.Phi_T_i[clustnum]= temp.t().i();
	flag = 0;
      } catch (NPDException) {
	flag++;
	if (flag >= ITERTRY) {
	  std::cout << "SamplePhi failed due to singular matrix after 10 tries" << endl;
	  exit(1);
	}
      }
    } while (flag >0);
  }
  
  if(mcsamplealpha)
    {
      double alphaj = sampleAlpha(result.pV[clustnum].Store(),prior.ee,prior.ff,mt);
      //std::cout << alphaj << endl;
      if (alphaj < 0.0001) { 
	alphaj = 0.0001;
      }
      result.alpha[clustnum] = alphaj;
    }
  
  return true;
}


bool CDP::iterate(CDPResult& result, MTRand& mt) { 
  //sample w
#if defined(CDP_TBB)
  //parallel_for(blocked_range<size_t>(0, prior.N),WSampler(this, &result),auto_partitioner());	
  parallel_for(blocked_range<size_t>(0, prior.N, 5000), WSampler(this, &result));
#else
  RowVector row(prior.D);
  for (int i = 0; i < prior.N;i++) {
    for (int j =0; j < prior.D; j++) {
      row[j] = mX[i][j];
    }
    if(mcsamplew)
      result.W[i] = sampleW(row,result.q,result.p,result.mu,result.Sigma_T_i,result.Sigma_log_det,mt);
    int index = result.W[i];
    if(mcsamplek)
      result.K[i] = sampleK(row,result.p[index],result.mu,result.Sigma_T_i,index,result.Sigma_log_det,mt);
  }
#endif
  
  //sample q
  //J by 1
  RowVector p;
  RowVector pV;
  if(mcsampleq)
    {
      sampleP(result.W,prior.N,result.alpha0,prior.J,p,pV,mt);
      result.q = p;
      result.qV = pV;
    }
  
  //sample alpha0
  if(mcsamplealpha0)
    {
      double alpha0 = sampleAlpha(result.qV.Store(),prior.e0,prior.f0,mt);
      //      std::cout << "alpha0=" << alpha0 << endl;
      if (alpha0 < 0.0001) {
	
	alpha0 = 0.0001;
      }
      result.alpha0 = alpha0;
    }

  // sample all parameters associated with each cluster DP mixture
  vector<vector<int> >w1d;
  partition(result.W,w1d);
#if defined(CDP_TBB)
  //std::cout << "MT iterate" << endl;
  parallel_for(blocked_range<size_t>(0, prior.J, 1), ClusterSampler(this, &result,&w1d));			
  //parallel_for(blocked_range<size_t>(0, prior.J), ClusterSampler(this, &result),auto_partitioner());		    
#else
  for(int j=0;j<prior.J;j++) {
    clusterIterate(j,w1d[j],result,mt);
  } 
#endif 
  
  return true;
}

//#if !defined(CDP_MPI)
//int main(int argc,char* argv[]) {
//#if defined(CDP_TBB)
//  task_scheduler_init init;
//#endif
//  Model model;
//  CDP cdp;
//  MTRand mt;
//  string pfile = "parameters.txt";
//  if (argc > 1) {
//    pfile = string(argv[1]);
//    if (pfile == "-DEFAULT" || pfile == "-Default" || pfile == "-default") {
//      model.Save("default.parameters.txt");
//      std::cout << "Default Parameters are saved in 'default.parameters.txt'" << endl;
//      exit(0);
//    }
//  }
//  if (!model.Load(pfile)) {
//    std::cout << "Loading parameters file failed" << endl;
//    exit (1);
//  }
//  
//  mt.seed(model.mnSeed);
//  cdp.LoadData(model);
//  cdp.prior.Init(model);
//  cdp.InitMCMCSteps(model);
//  
//  //purely for optimization
//  cdp.precalculate = cdp.msf.gammaln((cdp.prior.nu + (double)model.mnD)/2) - 
//    cdp.msf.gammaln(cdp.prior.nu/2) -0.5 * (double)model.mnD * log(cdp.prior.nu) - 0.5 * (double)model.mnD * 1.144729885849400;
//  
//  CDPResult result(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
//  
//  cdp.SimulateFromPrior2(result,mt);
//  
//  // if any values are to be loaded from file, load them here
//  cdp.LoadInits(model,result, mt);
//  
//  // see if we're dealing with a special case of J==1
//  cdp.CheckSpecialCases(model,result);
//  
//  // main mcmc loop
//  for (int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
//    std::cout << "it = " << (it+1) << endl;
//    cdp.iterate(result,mt);
//    
//    if (it >= model.mnBurnin) {
//      result.SaveDraws();
//      result.UpdateMeans();
//    }
//    
//  }
//  
//  // save final parameter values
//  result.SaveFinal();
//  // save posterior means for mixture component parameters
//  if(model.mnIter>0){
//    result.SaveBar();
//  }
//  
//  std::cout << "Done" << endl;
//  return 0;
//}
//#endif
