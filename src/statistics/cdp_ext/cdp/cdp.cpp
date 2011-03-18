/* cdp.cpp
 * @author Quanli Wang, quanli@stat.duke.edu
 */
#include <math.h>
#include <cstdio>
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
#include "cdpresultem.h"
#include "cdp.h"
#include "cdp_em.h"

#if defined(CDP_TBB)
	#include "cdpt.h"
#endif

#if defined(CDP_CUDA)
	#include "CDPBaseCUDA.h"
#if !defined(PYWRAP)
	#include <cutil_inline.h>
#endif
	#include <cuda_runtime_api.h>
#endif
int DIM,MEAN_CHD_DIM,PACK_DIM,CHD_DIM,LOGDET_OFFSET,DATA_PADDED_DIM,NCHUNKSIZE;

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
  mcsampleEta=true;

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
	mcsampleEta = model.sampleEta;
  }

// creates reverse lookup vectors wk2d: observations associated with components 1..JT
// and w1d: observations associated with clusters 1..J
void CDP::partition(int* W, vector<vector<int> >& w1d)
{
  w1d.clear();
  int i;
  //  int nT = prior.T;
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

// creates the objects KJ and KJunique that store the labels of the nonempty components within
// the cluster j.  KJ has duplicities and KJUnique does not.  In other words
// KJ[j] = 1 1 1 1 3 3 3 3 (the component membership for observations in cluster j)
// KJUnique[j] = 1 3 (the nonempty components in cluster j)
// only list the component membership for a single cluster characterized by w1d
void CDP::UpdateKJ(int* K,vector<int>& w1d, vector<int>& KJ){
  KJ.clear();
  //  KJUnique.clear();
  if ((int) w1d.size()> 0)
    {
      vector<int>::iterator it;
      for(it=w1d.begin(); it != w1d.end(); ++it)
	{
	  KJ.push_back(K[*it]);
	  //	  KJUnique.insert(K[*it]);
	}
    }
}

bool CDP::SimulateFromPrior2(CDPResult& result, MTRand& mt, int isEM) {
  int j,t;
  LowerTriangularMatrix temp;
	RowVector tmpEta(prior.T);
  //J
  //result.alpha0 = 1;
  /*
  LowerTriangularMatrix ltcov = Cholesky(prior.Phi0);
  SymmetricMatrix lambda = prior.Lambda0 / prior.nu0;
  //  LowerTriangularMatrix ltlambda = Cholesky(lambda);
  for (j = 0; j < prior.J; j++) {
    // m_j ~ N(m0,Phi0)
    result.m.push_back(msf.mvnormrand(prior.m0,ltcov,mt));
    // Phi_j ~ W(nu0,Lambda0/nu0)
    //    result.Phi.push_back(msf.wishartrand((int)prior.nu0,ltlambda,mt));
    result.Phi.push_back(prior.Lambda0);

    lambda = result.Phi[j]*prior.nu; // resuse lambda here
    temp << Cholesky(lambda);
    // log_determinant of (Phi_j*nu)
    result.Phi_log_det.push_back(msf.logdet(temp)); //used in evaluating inv wishart densities
        // inverse of transpose of Chol(nu*Phi_j)'
    result.Phi_T_i.push_back(temp.t().i());  //used in evaluating inv wishart densities

    // alpha_j ~ Ga(ee,ff)
    //result.alpha[j] = 1;
	result.alpha[j] = msf.gammarand(prior.ee,1.0/prior.ff,mt);
  }*/

  //J by T
  //for (j = 0; j < prior.J;j++) {
	
	// alpha_j ~ Ga(ee,ff)
	result.alpha[0] = msf.gammarand(prior.ee,1.0/prior.ff,mt);
	
	result.Phi.push_back(prior.Phi0);
	result.m.push_back(prior.m0);
	
	SymmetricMatrix phi = prior.Phi0 * prior.nu;
	phi = phi.i();
	LowerTriangularMatrix ltphi = Cholesky(phi);
	for (t = 0; t < prior.T;t++) {
		//int index = result.GetIndex(j,t);
		
		// eta_t ~ Ga(aa/2,aa/2);
		tmpEta[t] = msf.gammarand(prior.aa / 2.0, 2.0 / prior.aa,mt);
	   
		LowerTriangularMatrix temp1 = ltphi*(1.0 / sqrt(tmpEta[t]));
		
		// Sigma_t ~ IW(nu+2,nu*eta[t]*Phi0)
		result.Sigma.push_back(msf.invwishartrand((int)(prior.nu + 2), temp1,mt));
		//result.Sigma.push_back(prior.Lambda0);

		temp << Cholesky(result.Sigma[t]);
		// log of determinant of Chol(Sigma_j,t)
		result.Sigma_log_det.push_back(msf.logdet(temp));
		// inverse of transpose of Chol(Sigma_j,t)
		result.L_i.push_back(temp.i());

		SymmetricMatrix temp2 = result.Sigma[t] *prior.gamma;

		// mu_j ~ N(m_j,gamma*Sigma_t)
		result.mu.push_back(msf.mvnormrand(prior.m0,temp2,mt));
	}

    result.eta.push_back(tmpEta);
    RowVector p(prior.T);p=1./(double)prior.T;
    RowVector pV(prior.T);pV=1/(double)prior.T;
    pV[prior.T-1] = 1.;
    // int dummy;
    //    sampleP(&dummy,0,result.alpha[j],prior.T,p,pV,mt);
    result.p.push_back(p);
    result.pV.push_back(pV);
  //}

  //J by 1
  //  sampleP(&dummy,0,result.alpha0,prior.J,p,pV,mt);
  result.q = 1./(double) prior.J;
  result.qV = 1/(double) prior.J;
  result.qV[prior.J-1] = 1.;

  if (isEM ==0) {
	#if defined(CDP_CUDA)
		cuda.sampleWK(result.q,result.p,result.mu,result.L_i, result.Sigma_log_det,mt,result.W,result.K,result.Z);
	#else
		for (int i = 0; i < prior.N;i++) {
			RowVector row(prior.D);
			for (j =0; j < prior.D; j++) {
				row[j] = mX[i][j];
			}
			sampleWK(row,result.q,result.p,result.mu,result.L_i,result.Sigma_log_det,result.W[i],result.K[i],mt);
		}
	#endif
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
	
	//Are we going to sample eta?
	if (!model.sampleEta) {
		for (int j=0; j<prior.J; j++) {
			for (int t=0; t<prior.T; t++) {
				result.eta[j][t] = 1.0;
			}
		}
	}

}


// sampling steps for a single layer DP mixture:
bool CDP::clusterIterate_one(CDPResult& result, MTRand& mt)
{
	RowVector PostMu;
    SymmetricMatrix PostSigma;
    LowerTriangularMatrix li;
    double logdet;
	
	Matrix sigmaInv(prior.D, prior.D);
	//Matrix workMat(prior.D, prior.D);

	vector<int> w1d;
	int i,j,k,t;
	for (i = 0; i < prior.N; i++) {
		w1d.push_back(i);
	}

	int dim = prior.D * (prior.D + 1) / 2;
	double *clustercov = new double[prior.T * dim];
	double *clustermean = new double[prior.T * prior.D];
	double *clustercount = new double[prior.T]; 
	
#if defined(CDP_MEANCOV)
	UpperTriangularMatrix temp(prior.D); temp = 0;
	LowerTriangularMatrix temp2(prior.D); temp2 = 0;
	for (t = 0; t < prior.T;t++) {
		clustercount[t] = cuda.hClusterCount[t];

		int offset = t  *prior.D;
		double *pmean = clustermean + offset;
		for (j = 0; j < prior.D; j++) {	
			*pmean++ = cuda.hMean[offset + j];
		}

		//cuda.hCov is saved as an upper triangular matrix, T by D*D
		//pcov is saved as a lower triangular matrix, T by dim
		offset = t * dim;
		int ibase = t * prior.D * prior.D;
		double *pcov = clustercov + offset;
		double *ptemp = temp.Store();
		for (j = 0; j < dim;j++) {
			ptemp[j] = cuda.hCov[ibase+j];
		}
		LowerTriangularMatrix temp2 = temp.t();
		double *ptemp2 = temp2.Store();
		for (j = 0; j < dim; j++) {
			//std::cout << pcov[j] << "\t" << ptemp2[j] << endl;
			pcov[j] = ptemp2[j];
		}
	}
#else
	memset(clustermean,0,sizeof(double) * prior.T * prior.D);
	memset(clustercount,0,sizeof(double) * prior.T);
	memset(clustercov,0,sizeof(double) * prior.T * dim);
	//update sums
	for (i = 0; i < prior.N; i++) {
		int index = result.K[i];
		//std::cout << index << endl;
		double *pmean = clustermean + index * prior.D;
		double	*pX = mX[i];
		for (j = 0; j < prior.D; j++) {
			*pmean++ += *pX++;
		}
		pX -= prior.D;
		double *pcov = clustercov + index * dim;
		for (j = 0; j < prior.D; j++) {
			for (k = 0; k <=j; k++) {
				*pcov++ += pX[j] * pX[k];
			}
		}
		clustercount[index] += 1.0;
	}
#endif
	//caculate covariance matrix
	result.r = 0;
	for (t = 0; t < prior.T;t++) {
		double *pcov = clustercov + t * dim;
		double *pmean = clustermean + t * prior.D;
		if (clustercount[t] > 1) {	
			for (j = 0; j < prior.D; j++) {
				*pmean++ /= clustercount[t];
			}
			pmean -= prior.D;
			for (j = 0; j < prior.D; j++) {
				for (k = 0; k <=j; k++) {
					*pcov++ -= clustercount[t] * pmean[j] * pmean[k]; 
				}
			}
			result.r = result.r + 1;
		} else if(clustercount[t] > 0)  {
			memset(pcov,0,sizeof(double)*dim);
			result.r = result.r + 1;
		} else {
			//do nothing
		}
	}


	SymmetricMatrix mycov(prior.D);
	RowVector xbar(prior.D);
	double postnu;
	double postgamma;
	double gamma1,gamma2;
	SymmetricMatrix postS(prior.D);
	RowVector postm;
	for (t=0; t < prior.T;t++) {
		mycov = 0.0; xbar = 0.0;
		double *pmean = clustermean + t * prior.D;
		double *pcov =  clustercov + t * dim;
		memcpy(xbar.Store(),pmean,sizeof(double)*prior.D);
		memcpy(mycov.Store(),pcov,sizeof(double)*dim);
		i=0;
		do{
		
			if(mcsampleEta){
				if (clustercount[t]>0) {
					sigmaInv << result.L_i[t].t() * result.L_i[t];
					//workMat = result.Phi[0] * sigmaInv;
					gamma1 = (prior.aa + prior.D*(prior.nu + (double)prior.D + 1.0)) / 2.0;
					gamma2 = (prior.aa + prior.nu*(result.Phi[0] * sigmaInv).Trace()) / 2.0;	
				} else {
					gamma1 = prior.aa / 2.0;
					gamma2 = prior.aa / 2.0;
				}
			//std::cout << "gamma 1 " << gamma1 << endl;
			//std::cout << "gamma 2 " << gamma2 << endl;

				result.eta[0][t] = msf.gammarand(gamma1, 1.0/gamma2,mt);
				if (result.eta[0][t] < 10e-4) {
					result.eta[0][t] = 10e-4;
				}
			//std::cout << "eta " << result.eta[0][t] << endl;
			}
			double nuprime = prior.nu*result.eta[0][t];
			if (clustercount[t] >0) {
				double temp = clustercount[t] * prior.gamma+1.0;
				postnu = prior.nu + 3.0 + clustercount[t];
				RowVector rtemp = (xbar-result.m[0]);
				postS << mycov+rtemp.t() * rtemp/(clustercount[t] / temp) + result.Phi[0]*nuprime ;
				postm = (result.m[0]+xbar *(clustercount[t] * prior.gamma)) / temp;
				postgamma = prior.gamma / temp;
			} else {
				postnu = prior.nu+2.0;
				postS << result.Phi[0]*nuprime;	
				postm = result.m[0];
				postgamma = prior.gamma;
			}
			//std::cout << postS << endl;
			int flag = 0;
			int ITERTRY = 10;
			do {	
				SymmetricMatrix mySinv(prior.D); mySinv = 0.0;
				try {
					mySinv << postS.i();
					//std::cout << mySinv << endl;		problems here,exception not caught
					LowerTriangularMatrix mySinvL = Cholesky(mySinv);
					PostSigma = msf.invwishartrand((int)postnu,mySinvL,mt);
					//std::cout << "invwishartrand" << endl;
					LowerTriangularMatrix mycovinvL = Cholesky(PostSigma);
					logdet = msf.logdet(mycovinvL);
					li = mycovinvL.i();
					mycovinvL << mycovinvL * sqrt(postgamma);
					PostMu = msf.mvnormrand(postm, mycovinvL,mt);
					flag = 0;
				} catch (...) {
					flag++;
					//std::cout << mySinv << endl;		//problems here,exception not caught
					//postS = postS + result.Phi[0]*prior.nu; 
					postS = result.Phi[0]*prior.nu; 
					if (flag >= ITERTRY) {
						std::cout << "SampleMuSigma failed due to singular matrix after 10 tries" << endl;
						exit(1);
					}
				}
			} while	(flag >0);
			if(mcsamplemu) {
				result.mu[t] = PostMu;
			}
			if(mcsampleSigma) {
				result.Sigma[t] = PostSigma;
				result.Sigma_log_det[t] = logdet;
				result.L_i[t] = li;
			}
			i++;
		} while (i<10 && mcsampleEta);
	}
	
	delete [] clustercov;
	delete [] clustermean;
	delete [] clustercount;
	
  //sample p
  vector<int> KJ;
  UpdateKJ(result.K,w1d,KJ); // KJ is a list of component numbers for each observation in this cluster
	
  RowVector p;
  RowVector pV;
  if(mcsamplep)
    {
      int n = (int)w1d.size();
      sampleP(KJ,n,result.alpha[0],prior.T,p,pV,mt);
      result.p[0] = p;
      result.pV[0] = pV;
    }

  //alpha

  double alphaj = sampleAlpha(result.pV[0].Store(),prior.ee,prior.ff,mt);
  //std::cout << alphaj << endl;
  if (alphaj < 0.0001) {
		alphaj = 0.0001;
  }
  result.alpha[0] = alphaj;


  return true;
}

bool CDP::iterate(CDPResult& result, MTRand& mt) {
  //sample w
#if defined(CDP_TBB)
  //parallel_for(blocked_range<size_t>(0, prior.N),WSampler(this, &result),auto_partitioner());
  parallel_for(blocked_range<size_t>(0, prior.N, 5000), WSampler(this, &result));
#else
#if defined(CDP_CUDA)
  cuda.sampleWK(result.q, result.p, result.mu, result.L_i, result.Sigma_log_det, mt, result.W, result.K, result.Z);
#else
  RowVector row(prior.D);
	for (int i = 0; i < prior.N; i++) {
		for (int j = 0; j < prior.D; j++) {
			row[j] = mX[i][j];
		}
		if (mcsamplew && mcsamplek) {
			sampleWK(row, result.q, result.p, result.mu, result.L_i,
					result.Sigma_log_det, result.W[i], result.K[i], mt);
		} else if (mcsamplew && !mcsamplek) {
			result.W[i] = sampleW(row, result.q, result.p, result.mu,
					result.L_i, result.Sigma_log_det, mt);
		} else if (mcsamplek && !mcsamplew) {
			result.K[i] = sampleK(row, result.Z+i ,result.p[result.W[i]], result.mu,
					result.L_i, result.W[i], result.Sigma_log_det, mt);
		} 
	}
#endif
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
      if (alpha0 < 0.0001) {
	alpha0 = 0.0001;
      }
      result.alpha0 = alpha0;
    }
  
  if (prior.J == 1) {
	clusterIterate_one(result,mt);
  } else {
		//two layer model is disabled here
  }
  return true;
}




int main(int argc,char* argv[]) {
#if defined(CDP_TBB)
  task_scheduler_init init;
#endif
  
  Model model;
  MTRand mt;
  string pfile = "parameters.txt";
  if (argc > 1) {
    pfile = string(argv[1]);
    if (pfile == "-DEFAULT" || pfile == "-Default" || pfile == "-default") {
      model.Save("default.parameters.txt");
      std::cout << "Default Parameters are saved in 'default.parameters.txt'" << endl;
      exit(0);
    }
  }
  if (!model.Load(pfile)) {
    std::cout << "Loading parameters file failed" << endl;
    exit (1);
  }

  
  mt.seed(model.mnSeed);
  if (model.mstralgorithm == "bem") {
	  //std::cout << "Running bem" << endl;
	  CDP_EM cdp;
	  cdp.EMAlgorithm(model,mt);
  } else {
	  CDP cdp;
	  cdp.LoadData(model);
	  cdp.mcRelabel = model.Zrelabel;
	  
	  cdp.prior.Init(model);
	  cdp.InitMCMCSteps(model);

	  //purely for optimization
	//   cdp.precalculate = cdp.msf.gammaln((cdp.prior.nu + (double)model.mnD)/2) -
	//     cdp.msf.gammaln(cdp.prior.nu/2) -0.5 * (double)model.mnD * log(cdp.prior.nu) - 0.5 * (double)model.mnD * 1.144729885849400;

	  CDPResult result(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
	  result.OpenPostFiles();

	#if defined(CDP_CUDA)
		if (model.mnGPU_Sample_Chunk_Size < 0) {
				NCHUNKSIZE = model.mnN;
		} else {
				NCHUNKSIZE = model.mnGPU_Sample_Chunk_Size;
		}
		cdp.cuda.initializeInstance(model.startDevice,cdp.prior.J,cdp.prior.T, cdp.prior.N, 
						cdp.prior.D,model.numberDevices); 
		cdp.cuda.initializeData(cdp.mX);
	#endif

	  cdp.SimulateFromPrior2(result,mt,0);

	  // if any values are to be loaded from file, load them here
	  cdp.LoadInits(model,result, mt);

	  //just to agree with Dan's code, should be fixed later, here we are initilizing K twice when K is not loaded
	  if(!model.loadK && cdp.prior.J == 1) { 
		  for (int tt = 0; tt < cdp.prior.N;tt++) { //skip N random numbers, for test only, should be removed in release
			  double temp = mt();
		  }
		  #if defined(CDP_CUDA)
		  cdp.cuda.sampleWK(result.q, result.p, result.mu, result.L_i, result.Sigma_log_det, mt, result.W, result.K, result.Z);
		  #else
		  //call sampleK instead
		  #endif

	  }

	  // see if we're dealing with a special case of J==1
	  cdp.CheckSpecialCases(model,result);

	  #if defined(CDP_CUDA) && !defined(PYWRAP)
	    unsigned int hTimer;
		unsigned int hRelabelTimer;

		cutilCheckError(cutCreateTimer(&hRelabelTimer));
		cutilCheckError(cutResetTimer(hRelabelTimer));
		cutilCheckError(cutCreateTimer(&hTimer));
		cutilCheckError(cutResetTimer(hTimer));
		cutilCheckError(cutStartTimer(hTimer));
	  #else
		time_t tStart, tEnd;
		//long tStart, tEnd;
		//tStart = clock();
		time(&tStart);
	  #endif
	
	  // main mcmc loop
	  for (unsigned int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
		 int printout = model.mnPrintout > 0 && it % model.mnPrintout == 0 ? 1: 0;
		if (printout>0) {  
			std::cout << "it = " << (it+1) << endl;
		}
	
		cdp.iterate(result,mt);

		  if (model.Zrelabel) {
			#if defined(CDP_CUDA) && !defined(PYWRAP)
				cutilCheckError(cutStartTimer(hRelabelTimer));
			#endif
			cdp.ComponentRelabel(result);
			#if defined(CDP_CUDA) && !defined(PYWRAP)
				cutilCheckError(cutStopTimer(hRelabelTimer));
			#endif
		  }

		if (it >= model.mnBurnin) {
		  result.SaveDraws();
		  result.UpdateMeans();
		}
	  }
	  if (model.mnPrintout >0) {
		  #if defined(CDP_CUDA) && !defined(PYWRAP)
			cutilCheckError(cutStopTimer(hTimer));
			printf("GPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));
			printf("Relabeling Processing time: %f (ms) \n", cutGetTimerValue(hRelabelTimer));
		  #else
			//tEnd = clock();
		  time(&tEnd);
			cout << "time lapsed:" << fabs(difftime(tEnd,tStart))  << "seconds"<< endl;
		  #endif
	  }

	  
	  // save final parameter values
	  result.SaveFinal();
	  // save posterior means for mixture component parameters
	  if(model.mnIter>0){
		result.SaveBar();
	  }
  }
  if (model.mnPrintout >0) {
	std::cout << "Done" << endl;
  }
  return 0;
}
