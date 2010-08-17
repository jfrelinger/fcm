#include "newmatap.h"
#include "newmatio.h"
#include "cdpresult.h"
#include "cdpresultem.h"


CDPResultEM::~CDPResultEM(void)
{
	/*delete [] W;
	delete [] K;
	postmufile.close();
	postpfile.close();
	postSigmafile.close();
	postmfile.close();
	postqfile.close();
	postPhifile.close();*/
	pi_ij.CleanUp();
	sumiPi.CleanUp();
	xbar.CleanUp();
//	maxPdf.CleanUp();
	
}
