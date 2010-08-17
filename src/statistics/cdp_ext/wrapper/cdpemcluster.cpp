//#include "MersenneTwister.h"
//#include "Model.h"
//#include "cdp.h"

#include "cdpemcluster.h"

#if defined(CDP_CUDA)
#include "CDPBaseCUDA.h"
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#endif


cdpemcluster::~cdpemcluster() {
	if (resultInit) {
		if (param != 0){
			delete param;
			param = 0;
		};
	};
}

//cdpemcluster::cdpemcluster(int d, int n, double** x, int iter, int burn) {
cdpemcluster::cdpemcluster(int n, int d, double* x) {
  
  model.mstralgorithm = "bem";
  model.mnN = n;
  model.mnD = d;

  
  model.mdm0 = RowVector(d);
  for(int i=0;i<d;i++) {
	   model.mdm0[i] = 0;
  }
  //model.mdm0=0;
  cdpfunctions.mX = new double*[n];
  int i,j;
  for (i=0;i<n;++i){
  	cdpfunctions.mX[i] = new double[d];
  	for (j=0;j<d;++j){
  		int pos = i*d+j;
  		cdpfunctions.mX[i][j] = x[pos];
  		//std::cout << cdp.mX[i][j] << " " <<(*(x+i)+j)  << std::endl;
  		//cdp.mX[i][j] = *(*(x+i)+j);
  	}
  }
  cdp.mX = cdpfunctions.mX;
  cdp.prior.Init(model);
  cdpfunctions.prior = cdp.prior;
  
  verbose = false;
  resultInit = false;
};

void cdpemcluster::run(){
	//CDP_EM cdp;
	//cdp.EMAlgorithm(model,mt);
	//void CDP_EM::EMAlgorithm(Model& model, MTRand& mt){
	
	
	
	double oldLL = 10000000000.0;
	
	double errTol;
	SymmetricMatrix sigmaInv;
	Matrix workMat;
	
	//cdpfunctions.LoadData(model);
	//cdp.mX = cdpfunctions.mX;
	//cdpfunctions.mX = cdp.mX;
	//mX = cdpfunctions.mX;

	
	//cdp.prior.Init(model);
	//cdpfunctions.prior = cdp.prior;
	
	//CDPResultEM result(prior.J,prior.T,prior.N,prior.D);
	param = new CDPResultEM(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
	resultInit = true;
	
#if defined(CDP_CUDA)
	
	if (model.mnGPU_Sample_Chunk_Size < 0) {
		
		NCHUNKSIZE = model.mnN;
		
	} else {
		
		NCHUNKSIZE = model.mnGPU_Sample_Chunk_Size;
		
	}
	

	cdp.emcuda.initializeInstanceEM(model.startDevice,cdp.prior.J,cdp.prior.T, cdp.prior.N, 
								  
								  cdp.prior.D,model.numberDevices); 

	cdp.emcuda.initializeDataEM(cdp.mX);
	
#endif
	cdpfunctions.SimulateFromPrior2((*param),mt,1);
	// if any values are to be loaded from file, load them here
	cdpfunctions.LoadInits(model,(*param), mt);	
	// see if we're dealing with a special case of J==1
	cdpfunctions.CheckSpecialCases(model,(*param));
	
	cdp.emupdateeta = model.sampleEta;
	
	(*param).etaEst = RowVector(cdp.prior.T);
	
	if (cdp.emupdateeta) {
		for (int t=0; t<cdp.prior.T; t++) {
			sigmaInv << (*param).L_i[t].t() * (*param).L_i[t];
			workMat = (*param).Phi[0]*sigmaInv;
			(*param).etaEst[t]=(cdp.prior.aa - cdp.prior.nu + 2) / (cdp.prior.aa + cdp.prior.nu*(workMat.Trace()));
		}
	} else {
		for (int t=0; t<cdp.prior.T; t++) {
			(*param).etaEst[t]=1.0;
		}
	}

	
#if defined(CDP_CUDA)
	
	unsigned int hTimer;
	
	cutilCheckError(cutCreateTimer(&hTimer));
	
	cutilCheckError(cutResetTimer(hTimer));
	
    cutilCheckError(cutStartTimer(hTimer));
	
#else
	
	time_t tStart, tEnd;
	
	//long tStart, tEnd;
	
	//tStart = clock();
	
	time(&tStart);
	
#endif
	// main EM loop
	errTol = log(1.0 + model.mnErrorPerTol);
	
	for (int it = 0; (it < model.mnIter) & (fabs((*param).postLL - oldLL) > errTol) ; it++) {
		
		oldLL = (*param).postLL;
		
		int printout = model.mnPrintout > 0 && it % model.mnPrintout == 0 ? 1: 0;
		if (verbose) {  
			std::cout << "it = " << (it+1) << endl;
		}
		cdp.iterateEM((*param),printout);	
	}
	
 if (verbose) {
	#if defined(CDP_CUDA)
		cutilCheckError(cutStopTimer(hTimer));
		printf("GPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));
	#else
		//tEnd = clock();
		time(&tEnd);
		cout << "time lapsed:" << difftime(tEnd,tStart)  << "seconds"<< endl;
	#endif	
  }
	
	
	
};


void cdpemcluster::setVerbose(bool verbosity) {
	verbose = verbosity;
}

// check
//void cdpemcluster::step(){
//    cdp.iterate((*param),mt);
//    (*param).UpdateMeans();
//};
//
//void cdpemcluster::stepburn(){
//	cdp.iterate(*param,mt);
//};

void cdpemcluster::setseed(int x){ model.mnSeed = x; };
int cdpemcluster::getn(){ return model.mnN; };
int cdpemcluster::getd(){ return model.mnD; };

void cdpemcluster::setlambda0( double lambda0 ) {
	 model.mdlambda0 = lambda0;
};

double cdpemcluster::getlambda0() {
	 return model.mdlambda0;
};

void cdpemcluster::setphi0( double phi0 ) {
	 model.mdphi0 = phi0;
};

double cdpemcluster::getphi0() {
	 return model.mdphi0;
};

void cdpemcluster::setnu0( double nu0 ) {
	 model.mdnu0 = nu0;
};

double cdpemcluster::getnu0() {
	 return model.mdnu0;
};

void cdpemcluster::setgamma( double gamma ) {
	 model.mdgamma = gamma;
};

double cdpemcluster::getgamma() {
	 return model.mdgamma;
};

void cdpemcluster::setnu( double nu ) {
	 model.mdnu = nu;
};

double cdpemcluster::getnu() {
	 return model.mdnu;
};

void cdpemcluster::sete0( double e0 ) {
	 model.mde0 = e0;
};

double cdpemcluster::gete0() {
	 return model.mde0;
};

void cdpemcluster::setf0( double f0 ) {
	 model.mdf0 = f0;
};

double cdpemcluster::getf0() {
	 return model.mdf0;
};

void cdpemcluster::setee( double ee ) {
	 model.mdee = ee;
};


double cdpemcluster::getee() {
	 return model.mdee;
};

void cdpemcluster::setaa( double aa ) {
	model.mdaa = aa;
};

double cdpemcluster::getaa() {
	return model.mdaa;
}; 

void cdpemcluster::setff( double ff ) {
	 model.mdff = ff;
};

double cdpemcluster::getff() {
	 return model.mdff;
};

void cdpemcluster::setT(int t) { // top level clusters
	model.mnT = t;
}

int cdpemcluster::getT() {
	return model.mnT;
};


int cdpemcluster::getK(int idx) { // fetch k
	return (*param).K[idx];
}

void cdpemcluster::getK(int d, int* res) {
	for(int i = 0; i< d; i++) {
		res[i] = (*param).K[i];
	};
};	

void cdpemcluster::setJ(int j) { // compoenent clusters
	model.mnJ = j;
};

int cdpemcluster::getJ() {
	return model.mnJ;
};

void cdpemcluster::setBurnin(int t) {
	model.mnBurnin = t;
};

int cdpemcluster::getBurnin() {
	return model.mnBurnin;
};

void cdpemcluster::setIter(int t) {
	model.mnIter = t;
};

int cdpemcluster::getIter() {
	return model.mnIter;
};


// results
int cdpemcluster::getclustN(){ // is this needed? isn't it just model.mnJ * model.mnT?
	return (*param).J*(*param).T;
};

double cdpemcluster::getMu(int idx, int pos){
	return (*param).mu[idx][pos];
};

double cdpemcluster::getm(int idx, int pos){
	return (*param).m[idx][pos];
};

double cdpemcluster::getSigma(int i, int j, int k){
		return (((*param).Sigma.at(i)).element(k,j));
};

double cdpemcluster::getPhi(int i, int j, int k){
		return (((*param).Phi.at(i)).element(k,j));
};

double cdpemcluster::getp(int idx){ 
	int j = idx % (*param).J; // j
	int t = idx / (*param).J; // t
	return (*param).p[j][t]; // j by t for some reason.
};

void cdpemcluster::setgpunchunksize(int x){
	model.mnGPU_Sample_Chunk_Size = x;
}

int cdpemcluster::getgpunchunksize(){
	return model.mnGPU_Sample_Chunk_Size;
}

void cdpemcluster::setdevice(int x){
	model.startDevice = x;
}

int cdpemcluster::getdevice(){
	return model.startDevice;
}

int cdpemcluster::getnumberdevices(){
	return model.numberDevices;
}

void cdpemcluster::setnumberdevices(int x){
	model.numberDevices = x;
}

// turn samplers on/off
bool cdpemcluster::samplem(){
  return model.samplem;
}

void cdpemcluster::samplem(bool x){
  model.samplem = x;
}

bool cdpemcluster::samplePhi(){
  return model.samplePhi;
}

void cdpemcluster::samplePhi(bool x){
  model.samplePhi = x;
}

bool cdpemcluster::samplew(){
  return model.samplew;
}

void cdpemcluster::samplew(bool x){
  model.samplew = x;
}

bool cdpemcluster::sampleq(){
  return model.sampleq;
}

void cdpemcluster::sampleq(bool x){
  model.sampleq = x;
}

bool cdpemcluster::samplealpha0(){
  return model.samplealpha0;
}

void cdpemcluster::samplealpha0(bool x){
  model.samplealpha0 = x;
}

bool cdpemcluster::samplemu(){
  return model.samplemu;
}

void cdpemcluster::samplemu(bool x){
  model.samplemu = x;
}

bool cdpemcluster::sampleSigma(){
  return model.sampleSigma;
}

void cdpemcluster::sampleSigma(bool x){
  model.sampleSigma = x;
}

bool cdpemcluster::samplek(){
  return model.samplek;
}

void cdpemcluster::samplek(bool x){
  model.samplek = x;
}

bool cdpemcluster::samplep(){
  return model.samplep;
}

void cdpemcluster::samplep(bool x){
  model.samplep = x;
}

bool cdpemcluster::samplealpha(){
  return model.samplealpha;
}

void cdpemcluster::samplealpha(bool x){
  model.samplealpha = x;
}

void cdpemcluster::loadMu(int n, int d, double* x) {
	loadRowsCols(x, (*param).mu, n, d);
};

void cdpemcluster::loadm(int n, int d, double* x) {
	loadRowsCols(x, (*param).m, n, d);
};

void cdpemcluster::loadp(int n, int d, double* x) {
	loadRowsCols(x, (*param).p, n, d);
};

void cdpemcluster::loadpV(int n, int d, double* x) {
	loadRowsCols(x, (*param).pV, n, d);
};

void cdpemcluster::loadalpha0(double x) {
	(*param).alpha0 = x;
};

void cdpemcluster::loadSigma(int i, int j, int k, double* x) {
	loadRowsCols(x, (*param).Sigma, i, j, k);
};

void cdpemcluster::loadPhi(int i, int j, int k, double* x) {
	loadRowsCols(x, (*param).Phi, i, j, k);
};


void cdpemcluster::loadW(int i, double* x) {
	loadRows(x, (*param).W, i);
};

void cdpemcluster::loadK(int i, double* x) {
	loadRows(x, (*param).K, i);
};

void cdpemcluster::loadq(int i, double* x) {
	loadRows(x, (*param).q, i);
};

void cdpemcluster::loadqV(int i, double* x) {
	loadRows(x, (*param).qV, i);
};

void cdpemcluster::loadalpha(int i, double* x) {
	loadRows(x, (*param).alpha, i);
};

void cdpemcluster::loadRowsCols(double* from, vector<SymmetricMatrix>& to, int idx, int rows, int columns) {
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

void cdpemcluster::loadRows(double* from, int* to, int cols) {
	for(int i=0; i<cols;++i){
		to[i] = int(from[i]);
	};
};

void cdpemcluster::loadRowsCols(double* from, vector<RowVector>& to, int n, int d){
	for(int i=0;i<n;++i){
		to.push_back(RowVector(Real(d)));
		for(int j=0;j<d;++j){
			int pos = i*d+j;
			//to.push_back(from[pos]);
			to.at(i)[j] = from[pos];
		};
	}; 
};

void cdpemcluster::loadRows(double* from, RowVector& to, int cols){
	for(int i=0; i<cols;++i){
		to[i] = from[i];
	};
};

void cdpemcluster::printModel(){
	model.Print();
};
