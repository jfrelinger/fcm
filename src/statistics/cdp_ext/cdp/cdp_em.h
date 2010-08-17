#pragma once

#if  defined(PYWRAP)
	#include "cdpresult.h"
	#include "cdpresultem.h"
	#include "cdpbase.h"
	#include "cdp.h"
#endif

class Model;
class CDP_EM:public CDPBase
{
 public:
  CDP_EM();
  ~CDP_EM(void);

  bool iterateEM(CDPResultEM& result,int printout);
  void EMAlgorithm(Model& model, MTRand& mt);	


	
  // EM: functions for updating posterior parameters
  void getPi_ij(CDPResultEM& result);
  void getXbar(CDPResultEM& result);
  void updatePandAlpha(CDPResultEM& result);
  void updateMu(CDPResultEM& result);
  void updateSigma(CDPResultEM& result); 
	
  bool emupdateeta;
 private:
  CDP cdpfunctions; 
};
