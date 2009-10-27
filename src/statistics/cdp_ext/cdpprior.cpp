#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "Model.h"
#include "cdpprior.h"
CDPPrior::CDPPrior(void)
{
}

CDPPrior::~CDPPrior(void)
{
}
// void CDPPrior::SetAlphas(double de0, double df0, double dee, double dff) {
// 	e0 = de0;
// 	f0 = df0;
// 	ee = dee;
// 	ff = dff;
// }
// void CDPPrior::UpdateDefault() {
//   	gamma = sqrt(stn);
//   	m0 = RowVector(D);m0 = 0;
// 	Phi0 = SymmetricMatrix(D);Phi0 = 0;
// 	Lambda0 = SymmetricMatrix(D); Lambda0 = 0;
// 	for (int i = 0; i < D; i++) {
// 		Phi0[i][i] = stn;
// 		Lambda0[i][i] = Phi0[i][i] / stn / stn;
// 	}
// 	nu0 = D + 1;
// }
void CDPPrior::Init(Model& model) {
	J = model.mnJ;
	T = model.mnT;
	D = model.mnD;
	N = model.mnN;
	m0 = model.mdm0;
		
	nu = model.mdnu;
	nu0 = D + model.mdnu0;
	gamma = model.mdgamma;
	
	Phi0 = SymmetricMatrix(D);Phi0 = 0;
	Lambda0 = SymmetricMatrix(D); Lambda0 = 0;
	for (int i = 0; i < D; i++) {
	  Phi0[i][i] = model.mdphi0;
	  Lambda0[i][i] = model.mdlambda0;
	}
	
	e0 = model.mde0;
	f0 = model.mdf0;
	ee = model.mdee;
	ff = model.mdff;
	
	//	SetAlphas(model.mde0,model.mdf0,model.mdee,model.mdff);
}
