//#include "MersenneTwister.h"
//#include "Model.h"
//#include "cdp.h"

#include "cdpcluster.h"

//cdpcluster::cdpcluster(int d, int n, double** x, int iter, int burn) {
cdpcluster::cdpcluster(int n, int d, double* x) {
//  Model model;
//  CDP cdp;
//  MTRand mt;

  model.mnN = n;
  model.mnD = d;

  
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
};

void cdpcluster::setVerbose(bool verbosity) {
	verbose = verbosity;
}

void cdpcluster::run(){
  cdp.prior.Init(model);
  cdp.InitMCMCSteps(model);
  //purely for optimization
  cdp.precalculate = cdp.msf.gammaln((cdp.prior.nu + (double)model.mnD)/2) - 
    cdp.msf.gammaln(cdp.prior.nu/2) -0.5 * (double)model.mnD * log(cdp.prior.nu) - 0.5 * (double)model.mnD * 1.144729885849400;
  
  CDPResult result(cdp.prior.J,cdp.prior.T,cdp.prior.N,cdp.prior.D);
  
  cdp.SimulateFromPrior2(result,mt);
  
  // if any values are to be loaded from file, load them here
  cdp.LoadInits(model,result, mt);
  
  // see if we're dealing with a special case of J==1
  cdp.CheckSpecialCases(model,result);
  // main mcmc loop
  for (int it = 0; it < model.mnBurnin + model.mnIter ; it++) {
  	if(verbose) {
    	std::cout << "it = " << (it+1) << endl;
  	}
    cdp.iterate(result,mt);
    
    if (it >= model.mnBurnin) {
      //result.SaveDraws();
      result.UpdateMeans();
    }
    
  }
  
  // save final parameter values
  //result.SaveFinal();
  // save posterior means for mixture component parameters
  //if(model.mnIter>0){
    //result.SaveBar();
  //}
  if(verbose) { 
  	std::cout << "Done" << std::endl;
  }
  param = &result;
};
// model getters and setters
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
	return (*param).mu.size();
};

double cdpcluster::getMu(int idx, int pos){
	return (*param).mu[idx][pos];
};

double cdpcluster::getSigma(int i, int j, int k){
	//std::cout << i << j << k << std::endl;
	//double x;
	if (j <= k){
		//std::cout << ((*param).Sigma[i].element(j,k)) << std::endl;
		return ((*param).Sigma[i].element(j,k));
	} else {
		//std::cout << ((*param).Sigma[i].element(k,j)) << std::endl;
		return ((*param).Sigma[i].element(k,j));
	};

};

void cdpcluster::printSigma() {
	for (int i = 0; i < (*param).J*(*param).T; i++) {
		for (int j = 0; j < (*param).D; j++) {
			for (int k = 0; k <= j; k++) {
				//std::cout << "j<k: " << i << j << k << std::endl;
				std::cout << (*param).Sigma[i].element(j,k) << "\t";
			}
			for (int k = j+1; k < (*param).D; k++) {
				std::cout << (*param).Sigma[i].element(k,j) << "\t";
			}
		}
		std::cout << endl;
	}
};

void cdpcluster::printSigma(int i, int j, int k){
	if (j <= k){
		std::cout << (*param).Sigma[i].element(j,k);
	} else {
		std::cout << (*param).Sigma[i].element(k,j);
	};
	std::cout << std::endl;
};

double cdpcluster::getp(int idx){ 
	int j = idx % (*param).J; // j
	int t = idx / (*param).J; // t
	return (*param).p[j][t]; // j by t for some reason.
};
