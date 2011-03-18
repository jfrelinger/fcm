//#include "MersenneTwister.h"
//#include "Model.h"
//#include "cdp.h"

#include "cdpcluster.h"

#include <stdexcept>

#if defined(CDP_CUDA)
#include "CDPBaseCUDA.h"
#include <cuda_runtime_api.h>
#endif


cdpcluster::~cdpcluster() {
	if (resultInit) {
	  delete param;
	  param = NULL;
	};
};

cdpcluster::cdpcluster(int n, int d, double* x) {


  model.mnN = n;
  model.mnD = d;
  cdp.prior.N = n;

  
  model.mdm0 = RowVector(d);
  for(int i=0;i<d;i++) {
	   model.mdm0[i] = 0;
  }
  //model.mdm0=0;

  int burn = 100;
  int iter = 1000;
  model.mnBurnin = burn;
  model.mnIter = iter;

  mt.seed(model.mnSeed);
  //mt.seed(1138);
  //cdp.LoadData(model);
  cdp.mX = new double*[n];
  int i,j;
  for (i=0;i<n;++i){
  	cdp.mX[i] = new double[d];
  	for (j=0;j<d;++j){
  		int pos = i*d+j;
  		cdp.mX[i][j] = x[pos];
  		//std::cout << cdp.mX[i][j] << " " <<(*(x+i)+j)  << std::endl;
  		//cdp.mX[i][j] = *(*(x+i)+j);
  	}
  }
   
  //cdp.mX = &x;
  
  verbose = false;
  resultInit = false;
}

void cdpcluster::makeResult(){
  cdp.prior.Init(model);
  cdp.InitMCMCSteps(model);
 
  //CDPResult result(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
  param = new CDPResult(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);


#if defined(CDP_CUDA)
  if (model.mnGPU_Sample_Chunk_Size < 0) {
    NCHUNKSIZE = model.mnN;
  } else {
     NCHUNKSIZE = model.mnGPU_Sample_Chunk_Size;
  }
    cdp.cuda.initializeInstance(model.startDevice,cdp.prior.J,cdp.prior.T, cdp.prior.N, cdp.prior.D,model.numberDevices);
    cdp.cuda.initializeData(cdp.mX);
#endif

  
  resultInit = true;

};

void cdpcluster::setVerbose(bool verbosity) {
	verbose = verbosity;
}

void cdpcluster::run() {

  //purely for optimization
  //cdp.precalculate = cdp.msf.gammaln((cdp.prior.nu + (double)model.mnD)/2) - 
  //  cdp.msf.gammaln(cdp.prior.nu/2) -0.5 * (double)model.mnD * log(cdp.prior.nu) - 0.5 * (double)model.mnD * 1.144729885849400;

  if (!resultInit) {
  	makeResult();
  }

  cdp.SimulateFromPrior2((*param),mt,0);


  
  // if any values are to be loaded from file, load them here
  //cdp.LoadInits(model,(*param), mt);

  if(!model.loadK && cdp.prior.J == 1) {
    for (int tt = 0; tt < cdp.prior.N;tt++) { //skip N random numbers, for test only, should be removed in release
      double temp = mt();
    }
    #if defined(CDP_CUDA)
      cdp.cuda.sampleWK((*param).q, (*param).p, (*param).mu, (*param).L_i, (*param).Sigma_log_det, mt, (*param).W, (*param).K, (*param).Z);
    #else
      //call sampleK instead
    #endif

  }

  // see if we're dealing with a special case of J==1
  cdp.CheckSpecialCases(model,(*param));

    //time_t tStart, tEnd;
    //long tStart, tEnd;
    //tStart = clock();
    //time(&tStart);


  // main mcmc loop
  for (int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
    if(verbose) {
      std::cout << "it = " << (it+1) << endl;
    }
    cdp.iterate((*param),mt);
	if (model.Zrelabel) {
		cdp.ComponentRelabel((*param));
	  }


    
    if (it >= model.mnBurnin) {
      (*param).SaveDraws();
      (*param).UpdateMeans();
    }
    
  }


  
};

void cdpcluster::step(){
    cdp.iterate((*param),mt);
    (*param).UpdateMeans();
};

void cdpcluster::stepburn(){
	cdp.iterate(*param,mt);
};

void cdpcluster::setseed(int x){ model.mnSeed = x; };
int cdpcluster::getn(){ return model.mnN; };
int cdpcluster::getd(){ return model.mnD; };

void cdpcluster::setlambda0( double lambda0 ) {
	 model.mdlambda0 = lambda0;
};

double cdpcluster::getlambda0() {
	 return model.mdlambda0;
};

void cdpcluster::setphi0( double phi0 ) {
	 model.mdphi0 = phi0;
};

double cdpcluster::getphi0() {
	 return model.mdphi0;
};

void cdpcluster::setnu0( double nu0 ) {
	 model.mdnu0 = nu0;
};

double cdpcluster::getnu0() {
	 return model.mdnu0;
};

void cdpcluster::setgamma( double gamma ) {
	 model.mdgamma = gamma;
};

double cdpcluster::getgamma() {
	 return model.mdgamma;
};

void cdpcluster::setnu( double nu ) {
	 model.mdnu = nu;
};

double cdpcluster::getnu() {
	 return model.mdnu;
};

void cdpcluster::sete0( double e0 ) {
	 model.mde0 = e0;
};

double cdpcluster::gete0() {
	 return model.mde0;
};

void cdpcluster::setf0( double f0 ) {
	 model.mdf0 = f0;
};

double cdpcluster::getf0() {
	 return model.mdf0;
};

void cdpcluster::setee( double ee ) {
	 model.mdee = ee;
};


double cdpcluster::getee() {
	 return model.mdee;
};

void cdpcluster::setaa( double aa ) {
	model.mdaa = aa;
};

double cdpcluster::getaa() {
	return model.mdaa;
}; 

void cdpcluster::setff( double ff ) {
	 model.mdff = ff;
};

double cdpcluster::getff() {
	 return model.mdff;
};

void cdpcluster::setT(int t) { // top level clusters
	model.mnT = t;
}

int cdpcluster::getT() {
	return model.mnT;
};


int cdpcluster::getK(int idx) { // fetch k
	return (*param).K[idx];
}

void cdpcluster::getK(int d, int* res) {
	for(int i = 0; i< d; i++) {
		res[i] = (*param).K[i];
	};
};	

void cdpcluster::setJ(int j) { // compoenent clusters
	model.mnJ = j;
};

int cdpcluster::getJ() {
	return model.mnJ;
};

void cdpcluster::setBurnin(int t) {
	model.mnBurnin = t;
};

int cdpcluster::getBurnin() {
	return model.mnBurnin;
};

void cdpcluster::setIter(int t) {
	model.mnIter = t;
};

int cdpcluster::getIter() {
	return model.mnIter;
};


// results
int cdpcluster::getclustN(){ // is this needed? isn't it just model.mnJ * model.mnT?
	return (*param).J*(*param).T;
};

double cdpcluster::getMu(int idx, int pos){
	return (*param).mu[idx][pos];
};

double cdpcluster::getm(int idx, int pos){
	return (*param).m[idx][pos];
};

double cdpcluster::getSigma(int i, int j, int k){
		return (((*param).Sigma.at(i)).element(k,j));
};

double cdpcluster::getPhi(int i, int j, int k){
		return (((*param).Phi.at(i)).element(k,j));
};

double cdpcluster::getp(int idx){ 
	int j = idx % (*param).J; // j
	int t = idx / (*param).J; // t
	return (*param).p[j][t]; // j by t for some reason.
};

void cdpcluster::setgpunchunksize(int x){
#if defined(CDP_CUDA)
	model.mnGPU_Sample_Chunk_Size = x;
#else
	throw runtime_error("CUDA Not Supported");
#endif
}

int cdpcluster::getgpunchunksize(){
#if defined(CDP_CUDA)
	return model.mnGPU_Sample_Chunk_Size;
#else
	throw runtime_error("CUDA Not Supported");
	return -1;
#endif
}

void cdpcluster::setdevice(int x){
#if defined(CDP_CUDA)
	model.startDevice = x;
#else
	throw runtime_error("CUDA Not Supported");
#endif
}

int cdpcluster::getdevice(){
#if defined(CDP_CUDA)
	return model.startDevice;
#else
	throw runtime_error("CUDA Not Supported");
	return -1;
#endif
}

int cdpcluster::getnumberdevices(){
	return model.numberDevices;
}

void cdpcluster::setnumberdevices(int x){
	model.numberDevices = x;
}

// turn samplers on/off
bool cdpcluster::samplem(){
  return model.samplem;
}

void cdpcluster::samplem(bool x){
  model.samplem = x;
}

bool cdpcluster::samplePhi(){
  return model.samplePhi;
}

void cdpcluster::samplePhi(bool x){
  model.samplePhi = x;
}

bool cdpcluster::samplew(){
  return model.samplew;
}

void cdpcluster::samplew(bool x){
  model.samplew = x;
}

bool cdpcluster::sampleq(){
  return model.sampleq;
}

void cdpcluster::sampleq(bool x){
  model.sampleq = x;
}

bool cdpcluster::samplealpha0(){
  return model.samplealpha0;
}

void cdpcluster::samplealpha0(bool x){
  model.samplealpha0 = x;
}

bool cdpcluster::samplemu(){
  return model.samplemu;
}

void cdpcluster::samplemu(bool x){
  model.samplemu = x;
}

bool cdpcluster::sampleSigma(){
  return model.sampleSigma;
}

void cdpcluster::sampleSigma(bool x){
  model.sampleSigma = x;
}

bool cdpcluster::samplek(){
  return model.samplek;
}

void cdpcluster::samplek(bool x){
  model.samplek = x;
}

bool cdpcluster::samplep(){
  return model.samplep;
}

void cdpcluster::samplep(bool x){
  model.samplep = x;
}

bool cdpcluster::samplealpha(){
  return model.samplealpha;
}

void cdpcluster::samplealpha(bool x){
  model.samplealpha = x;
}

void cdpcluster::loadMu(int n, int d, double* x) {
	loadRowsCols(x, (*param).mu, n, d);
};

void cdpcluster::loadm(int n, int d, double* x) {
	loadRowsCols(x, (*param).m, n, d);
};

void cdpcluster::loadp(int n, int d, double* x) {
	loadRowsCols(x, (*param).p, n, d);
};

void cdpcluster::loadpV(int n, int d, double* x) {
	loadRowsCols(x, (*param).pV, n, d);
};

void cdpcluster::loadalpha0(double x) {
	(*param).alpha0 = x;
};

void cdpcluster::loadSigma(int i, int j, int k, double* x) {
	loadRowsCols(x, (*param).Sigma, i, j, k);
};

void cdpcluster::loadPhi(int i, int j, int k, double* x) {
	loadRowsCols(x, (*param).Phi, i, j, k);
};


void cdpcluster::loadW(int i, double* x) {
	loadRows(x, (*param).W, i);
};

void cdpcluster::loadK(int i, double* x) {
	loadRows(x, (*param).K, i);
};

void cdpcluster::loadq(int i, double* x) {
	loadRows(x, (*param).q, i);
};

void cdpcluster::loadqV(int i, double* x) {
	loadRows(x, (*param).qV, i);
};

void cdpcluster::loadalpha(int i, double* x) {
	loadRows(x, (*param).alpha, i);
};

void cdpcluster::loadref(int n, double* x) {
	int curclass;

	model.Zrelabel = true;
	(*param).refZ = new int[(*param).N];
	for(int i=0; i<n;++i)
	{
		(*param).refZ[i] = int(x[i]);
	}
	(*param).refZobs = new int[(*param).T];
	for (int i = 0; i<(*param).N; i++) {
		curclass = (*param).refZ[i];
		(*param).refZobs[curclass] = (*param).refZobs[curclass]+1;
	}
	(*param).loadZ = true;

}

void cdpcluster::loadRowsCols(double* from, vector<SymmetricMatrix>& to, int idx, int rows, int columns) {
	for(int i=0;i<idx;++i){
		to.push_back(SymmetricMatrix(Real(rows)));
		for(int j=0;j<rows;++j){
			for(int k=0;k<columns;++k){
				int pos = (i*rows*columns)+(j*columns)+k;
				to[i][j][k] = from[pos];
			};
		};
	};
};


void cdpcluster::loadRows(double* from, int* to, int cols) {
	for(int i=0; i<cols;++i){
		to[i] = int(from[i]);
	};
};

void cdpcluster::loadRowsCols(double* from, vector<RowVector>& to, int n, int d){
	for(int i=0;i<n;++i){
		to.push_back(RowVector(Real(d)));
		for(int j=0;j<d;++j){
			int pos = i*d+j;
			//to.push_back(from[pos]);
			to.at(i)[j] = from[pos];
		};
	}; 
};

void cdpcluster::loadRows(double* from, RowVector& to, int cols){
	for(int i=0; i<cols;++i){
		to[i] = from[i];
	};
};

void cdpcluster::printModel(){
	model.Print();
};
