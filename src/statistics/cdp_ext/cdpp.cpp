//#define CDP_MPI
#if defined(CDP_MPI)
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
#include "mpi.h"
#include "cdp.h"
#include "cdpp.h"


// For MPI Communication
#define SAMPLEW	1
#define SAMPLEK	2
#define SAMPLEMUSIGMA	2

#define WORKLENGTH 2
#define RESULTLENGTH 2
//global variables
static int	myrank;		// used to identify processor
Model model;
CDPP cdp;
CDPP::CDPP(void)
{

}

CDPP::~CDPP(void)
{
	delete [] wbuffer;
	delete [] kbuffer;
}
void CDPP::InitPBuffer(void) {
	wbuffer = new double[GetWBufferSize()];
	kbuffer = new double[GetKBufferSize()];
}
int CDPP::GetWBufferSize() {
	return prior.J * (prior.D + 2 + prior.D * (prior.D + 1));
}
int CDPP::GetKBufferSize() {
	return prior.T * prior.J * (prior.D + 2 + prior.D * (prior.D + 1) / 2);
}
int CDPP::GetMuSigmaBufferSize() {
	return 2 + prior.D + 1 + prior.D * (prior.D + 1);
}
void CDPP::PrepareWBuffer(CDPResult& result) {
	int nJ = prior.J;
	int nD = prior.D;
	int i,j;
	int nSize= nD * (nD+1) / 2;
	double *p = wbuffer;
	double *q = result.q.Store();
	for (i = 0; i < nJ;i++) {
		*p++ = *q++;
		double *m = result.m[i].Store();
		for (j = 0; j < nD; j++) {
			*p++ = *m++;
		}
		double * psi_t_i = result.Phi_T_i[i].Store();
		for (j = 0; j < nSize; j++) {
			*p++ = *psi_t_i++;
		}
		*p++ = result.Phi_log_det[i];
		double * psi = result.Phi[i].Store();
		for (j = 0; j < nSize; j++) {
			*p++ = *psi++;
		}
	}
}
void CDPP::PostWBuffer(CDPResult& result) { //reverse of preparewbuffer
	int nJ = prior.J;
	int nD = prior.D;
	int i,j;
	int nSize= nD * (nD+1) / 2;
	double *p = wbuffer;
	double *q = result.q.Store();
	for (i = 0; i < nJ;i++) {
		*q++ = *p++;
		double *m = result.m[i].Store();
		for (j = 0; j < nD; j++) {
			*m++ = *p++;
		}
		double * psi_t_i = result.Phi_T_i[i].Store();
		for (j = 0; j < nSize; j++) {
			*psi_t_i++ = *p++;
		}
		result.Phi_log_det[i] = *p++;
		double * psi = result.Phi[i].Store();
		for (j = 0; j < nSize; j++) {
			*psi++ = *p++;
		}
	}
}
void CDPP::PrepareKBuffer(CDPResult& result) {
	int nJ = prior.J;
	int nD = prior.D;
	int nT = prior.T;
	int i,j;
	int nSize= nD * (nD+1) / 2;
	double *b = kbuffer;
	
	for (i = 0; i < nJ;i++) {
		double* q = result.p[i].Store();
		for (j = 0; j < nT; j++) { 
			*b++ = *q++;
		}
	}
	for (i = 0; i < nJ * nT;i++) {
		double* pmu = result.mu[i].Store();
		for (j = 0; j < nD; j++) { 
			*b++ = *pmu++;
		}
		double* psigma_t_i = result.Sigma_T_i[i].Store();
		for (j = 0; j < nSize; j++) {
			*b++ = *psigma_t_i++;
		}
		*b++ =result.Sigma_log_det[i];
	}
}

void CDPP::PostKBuffer(CDPResult& result) {
	int nJ = prior.J;
	int nD = prior.D;
	int nT = prior.T;
	int i,j;
	int nSize= nD * (nD+1) / 2;
	double *b = kbuffer;
	for (i = 0; i < nJ;i++) {
		double* q = result.p[i].Store();
		for (j = 0; j < nT; j++) { 
			*q++ = *b++;
		}
	}
	for (i = 0; i < nJ * nT;i++) {
		double* pmu = result.mu[i].Store();
		for (j = 0; j < nD; j++) { 
			*pmu++ = *b++;
		}
		double* psigma_t_i = result.Sigma_T_i[i].Store();
		for (j = 0; j < nSize; j++) {
			*psigma_t_i++ = *b++ ;
		}
		result.Sigma_log_det[i] = *b++;
	}
}
void CDPP::BroardcastW(CDPResult& result) {
	if (myrank == 0) {
		PrepareWBuffer(result);
	}
	MPI_Bcast(wbuffer,GetWBufferSize(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	if (myrank > 0) {
		PostWBuffer(result);
	}
}
void CDPP::BroardcastK(CDPResult& result) {
	if (myrank == 0) {
		PrepareKBuffer(result);
	}
	MPI_Bcast(kbuffer,GetKBufferSize(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	if (myrank > 0) {
		PostKBuffer(result);
	}
}
int CDPP::SampleW(int index,CDPResult& result) {
	RowVector row(mX.Ncols());
	for (int j =0; j < mX.Ncols(); j++) {
		row[j] = mX[index][j];
	}
	return sampleW(row,result.q,result.m,result.Phi_T_i,prior.nu,prior.gamma,result.Phi_log_det);
}
int CDPP::SampleK(int index, int wi, CDPResult& result) {
	RowVector row(mX.Ncols());
	for (int j =0; j < mX.Ncols(); j++) {
		row[j] = mX[index][j];
	}
	return  sampleK(row,result.p[wi],result.mu,result.Sigma_T_i,wi,result.Sigma_log_det);
}
void CDPP::SampleMuSigma(int index, int j, CDPResult& result) {
	RowVector PostMu;
	SymmetricMatrix PostSigma;
	UpperTriangularMatrix uti;
	double logdet;
	sampleMuSigma(wk2d[index],prior.nu,prior.gamma,result.m[j],result.Phi[j],PostMu,PostSigma,uti,logdet);
	if(mcsamplemu)	{      
		result.mu[index] = PostMu;
	}
	if(mcsampleSigma)	{
		result.Sigma[index] = PostSigma;
		result.Sigma_log_det[index] = logdet;
		result.Sigma_T_i[index] = uti;
	}

}
void CDPP::PackUnpackMuSigmaBuffer(CDPResult& result, double* workresult, int flag) {
	int i,index,size;
	double* p = workresult+2;
	index = (int)workresult[0];
	size = prior.D * (prior.D + 1) / 2;
	if (flag > 0) { //pack
		double* m = result.mu[index].Store();
		for (i = 0; i < prior.D; i++) {
			*p++ = *m++;
		}
		double* s = result.Sigma[index].Store();
		for (i = 0; i < size; i++) {
			*p++ = *s++;
		}
		double* si = result.Sigma_T_i[index].Store();
		for (i = 0; i < size; i++) {
			*p++ = *si++;
		}
		*p++ = result.Sigma_log_det[index];
	} else { //unpack
		double* m = result.mu[index].Store();
		for (i = 0; i < prior.D; i++) {
			*m++ = *p++;
		}
		double* s = result.Sigma[index].Store();
		for (i = 0; i < size; i++) {
			*s++ = *p++;
		}
		double* si = result.Sigma_T_i[index].Store();
		for (i = 0; i < size; i++) {
			*si++ = *p++;
		}
		result.Sigma_log_det[index] = *p++;
	}
}
void CDPP::SampleAllW(CDPResult& result) {
	int i,ntasks;
	int jobsRunning = 1;
	int work[WORKLENGTH];
	double workresults[RESULTLENGTH];
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	if (myrank == 0) {
		for (i = 0; i < prior.N;i++) {	
			if (jobsRunning < ntasks) {
				work[0] = i;
				MPI_Send(&work, WORKLENGTH, MPI_INT,jobsRunning,SAMPLEW,MPI_COMM_WORLD); 
				jobsRunning++;
			} else {
				MPI_Recv(&workresults,RESULTLENGTH,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEW,MPI_COMM_WORLD,&status);     
				result.W[(int)workresults[0]] = (int)workresults[1];
				work[0] = i;
				MPI_Send(&work, WORKLENGTH, MPI_INT,status.MPI_SOURCE,SAMPLEW,MPI_COMM_WORLD); 
			}
		}
		// NOTE: we still have some work requests out that need to be
		// collected. Collect those results now!
		// loop over all the slaves
		for(int rank=1; rank<jobsRunning; rank++) {
			MPI_Recv(&workresults,RESULTLENGTH,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEW, MPI_COMM_WORLD,&status);
			result.W[(int)workresults[0]] = (int)workresults[1];
		}
		for (i = 1; i < ntasks;i++) {	
			MPI_Send(&work, WORKLENGTH, MPI_INT,i,0,MPI_COMM_WORLD); 
		}
	} else {
		MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		while (status.MPI_TAG == SAMPLEW) {
			workresults[0] = work[0];
			workresults[1] = SampleW(work[0],result);
			MPI_Send(&workresults,RESULTLENGTH,MPI_DOUBLE,0,SAMPLEW,MPI_COMM_WORLD);
			MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}
void CDPP::SampleAllK(CDPResult& result) {
	int i,ntasks;
	int jobsRunning = 1;
	int work[WORKLENGTH];
	double workresults[RESULTLENGTH];
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	if (myrank == 0) {
		for (i = 0; i < prior.N;i++) {	
			if (jobsRunning < ntasks) {
				work[0] = i;
				work[1] = result.W[i];
				MPI_Send(&work, WORKLENGTH, MPI_INT,jobsRunning,SAMPLEK,MPI_COMM_WORLD); 
				jobsRunning++;
			} else { 
				MPI_Recv(&workresults,RESULTLENGTH,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEK,MPI_COMM_WORLD,&status);     
				result.K[(int)workresults[0]] = (int)workresults[1];
				work[0] = i;
				work[1] = result.W[i];
				MPI_Send(&work, WORKLENGTH, MPI_INT,status.MPI_SOURCE,SAMPLEK,MPI_COMM_WORLD); 
			}
		}
		// NOTE: we still have some work requests out that need to be
		// collected. Collect those results now!
		// loop over all the slaves
		for(int rank=1; rank<jobsRunning; rank++) {
			MPI_Recv(&workresults,RESULTLENGTH,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEK, MPI_COMM_WORLD,&status);
			result.K[(int)workresults[0]] = (int)workresults[1];
		}
		for (i = 1; i < ntasks;i++) {	
			MPI_Send(&work, WORKLENGTH, MPI_INT,i,0,MPI_COMM_WORLD); 
		}
	} else {
		MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		while (status.MPI_TAG == SAMPLEK) {
			workresults[0] = work[0];
			workresults[1] = SampleK(work[0],work[1],result);
			MPI_Send(&workresults,RESULTLENGTH,MPI_DOUBLE,0,SAMPLEK,MPI_COMM_WORLD);
			MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void CDPP::SampleAllMuSigma(CDPResult& result) {
	int i,j,t,ntasks;
	int jobsRunning = 1;
	int work[WORKLENGTH];
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	int resultlength;
	resultlength = GetMuSigmaBufferSize();
	double* workresults = new double[resultlength];
	if (myrank == 0) {
		for (j = 0; j < prior.J;j++) {
			for (t = 0; t < prior.T;t++) {
				int index = result.GetIndex(j,t);
				work[0] = index;
				work[1] = j;
				if (jobsRunning < ntasks) {	
					MPI_Send(&work, WORKLENGTH, MPI_INT,jobsRunning,SAMPLEMUSIGMA,MPI_COMM_WORLD); 
					jobsRunning++;
				} else { 
					MPI_Recv(workresults,resultlength,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEMUSIGMA,MPI_COMM_WORLD,&status);   
					PackUnpackMuSigmaBuffer(result,workresults,0);
					MPI_Send(&work, WORKLENGTH, MPI_INT,status.MPI_SOURCE,SAMPLEMUSIGMA,MPI_COMM_WORLD); 
				}
			}
		}
		// NOTE: we still have some work requests out that need to be
		// collected. Collect those results now!
		// loop over all the slaves
		for(int rank=1; rank<jobsRunning; rank++) {
			MPI_Recv(workresults,resultlength,MPI_DOUBLE,MPI_ANY_SOURCE,SAMPLEMUSIGMA, MPI_COMM_WORLD,&status);
			PackUnpackMuSigmaBuffer(result,workresults,0);
		}
		for (i = 1; i < ntasks;i++) {	
			MPI_Send(&work, WORKLENGTH, MPI_INT,i,0,MPI_COMM_WORLD); 
		}

	} else {
		MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		while (status.MPI_TAG == SAMPLEMUSIGMA) {
			workresults[0] = work[0];
			workresults[1] = work[1];
			SampleMuSigma(work[0],work[1],result);
			PackUnpackMuSigmaBuffer(result,workresults,1);
			MPI_Send(workresults,resultlength,MPI_DOUBLE,0,SAMPLEMUSIGMA,MPI_COMM_WORLD);
			MPI_Recv(&work,WORKLENGTH,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		}
	}
	delete [] workresults;
	MPI_Barrier(MPI_COMM_WORLD);
}
bool CDPP::piterate(CDPResult& result) {
	int i,j,t;
	LowerTriangularMatrix temp;
	//sample w
	if(mcsamplew) {
		BroardcastW(result);	//broadcast W-related to all nodes
		SampleAllW(result);
	}
	//sample k
	if(mcsamplek) {
		BroardcastK(result);	//broadcast K-related to all nodes
		SampleAllK(result);
	}

	//broadcast W and K to all slave nodes
	MPI_Bcast(result.K,prior.N,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(result.W,prior.N,MPI_INT,0,MPI_COMM_WORLD);
	
	//sample mu and sigma
	//it might be a good idea to let all slave nodes update wk2d,w1d and KJ
	//if (myrank > 0) {
		partition(result.W,result.K, wk2d,w1d);
		UpdateKJ(result.K,w1d,KJ);
	//}
	if(mcsamplemu || mcsampleSigma) {
		SampleAllMuSigma(result);
	}

	if (myrank ==0) {
		//sample p
		RowVector p;
		RowVector pV;
	  
		if(mcsamplep) {      
			for (j = 0; j < prior.J;j++) {
				int n = (int)w1d[j].size(); 
				sampleP(KJ[j],n,result.alpha[j],prior.T,p,pV);
				result.p[j] = p;
				result.pV[j] = pV;
			}
		}

	  //sample q
	  //J by 1
	  if(mcsampleq)
		{
		  sampleP(result.W,prior.N,result.alpha0,prior.J,p,pV);
		  result.q = p;
		  result.qV = pV;
		}
	  
	  //sample m, Phi, alpha
	  RowVector newmu;
	  SymmetricMatrix newphi;
	  for (j = 0; j < prior.J; j++) {
		vector<RowVector> muj;
		vector<SymmetricMatrix> Sigmaj;
		//    int n = KJUnique[j].size(); // number of occupied components in cluster j
		int n = KJ[j].size(); // number of observations in cluster j
		if (n > 0) {
			// changing these two lines effectively weights the components by membership
			//      set<int>::iterator it;
			//      for(it=KJUnique[j].begin(); it != KJUnique[j].end(); ++it) {
			vector<int>::iterator it;
			for(it=KJ[j].begin(); it != KJ[j].end(); ++it) {
				muj.push_back(result.mu[result.GetIndex(j,*it)]); // the occupied component means
				Sigmaj.push_back(result.Sigma[result.GetIndex(j,*it)]); // the occupied component covariances
			}
		}

		if(mcsamplem) {
			sampleM(n,prior.gamma,muj,Sigmaj,prior.m0,prior.Phi0,newmu);
			result.m[j] = newmu;
		 }
		if(mcsamplePhi) {
			samplePhi(n,prior.nu,Sigmaj,prior.nu0,prior.Lambda0,newphi);
			result.Phi[j] = newphi;
			temp = Cholesky(newphi);
			temp << temp * sqrt(prior.nu*(prior.gamma+1)/(prior.nu+2.0));
			result.Phi_log_det[j] = msf.logdet(temp);
			result.Phi_T_i[j]= temp.t().i();
		}
		if(mcsamplealpha)
		  {
		double alphaj = sampleAlpha(result.pV[j].Store(),prior.ee,prior.ff);
		//std::cout << alphaj << endl;
		if (alphaj < 0.0001) { 
		  alphaj = 0.0001;
		}
		result.alpha[j] = alphaj;
		  }
	  }
	  
	  //sample alpha0
	  if(mcsamplealpha0)
		{
		  double alpha0 = sampleAlpha(result.qV.Store(),prior.e0,prior.f0);
		  //      std::cout << "alpha0=" << alpha0 << endl;
		  if (alpha0 < 0.0001) {
		
		alpha0 = 0.0001;
		  }
		  result.alpha0 = alpha0;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
  return true;
}
int main(int argc,char* argv[]) {
	MPI_Init(&argc, &argv);
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
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);	
	cdp.mt.seed(model.mnSeed+myrank*10); 
	cdp.LoadData(model);
	cdp.prior.Init(model);
	cdp.InitMCMCSteps(model);

	//purely for optimization
	cdp.precalculate = cdp.msf.gammaln((cdp.prior.nu + (double)model.mnD)/2) - 
	cdp.msf.gammaln(cdp.prior.nu/2) -0.5 * (double)model.mnD * log(cdp.prior.nu) - 0.5 * (double)model.mnD * 1.144729885849400;
	CDPResult result(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
	cdp.SimulateFromPrior(result);
	cdp.InitPBuffer();
	
	int ntasks;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	if (ntasks == 1) {
		std::cout << "sigle node mode" << endl;
		cdp.LoadInits(model,result);
		for (int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
			std::cout << "it = " << (it+1) << endl;
			cdp.iterate(result);
			if (it >= model.mnBurnin) {
				result.UpdateMeans();
			}
		}
		result.SaveFinal();
		// save posterior means for mixture component parameters
		if(model.mnIter>0){
			result.SaveBar();
		}
	} else {
		if (myrank == 0) {
			std::cout << "cluster mode" << endl;
			cdp.LoadInits(model,result);
		}
		for (int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
			if (myrank ==0) {
				std::cout << "it = " << (it+1) << endl;
			}
			cdp.piterate(result);
			if (myrank ==0) {
				if (it >= model.mnBurnin) {
					result.UpdateMeans();
				}
			}
		}
		if (myrank == 0) {
			result.SaveFinal();
			// save posterior means for mixture component parameters
			if(model.mnIter>0){
				result.SaveBar();
			}
		}
	}
	std::cout << "Done" << endl;
	MPI_Finalize();
	return 0;
}
#endif
