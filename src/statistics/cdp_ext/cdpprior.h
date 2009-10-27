#pragma once
class Model;
class CDPPrior
{
public:
	CDPPrior(void);
	~CDPPrior(void);
public:
	double nu; //Sigma_j,t ~ Inv Wishart(nu+2, nu Phi_j) so E(Sigma) = Phi_j
	double nu0; 
	double e0;
	double f0;
	double ee;
	double ff;
	double gamma;
	RowVector m0;
	SymmetricMatrix Phi0;
	SymmetricMatrix Lambda0;

	int T;	//num of components	
	int J;  // num of clusters
	int D;  // Dim 
	//	double stn;

	int N;	//number of points

	void SetAlphas(double de0, double df0, double dee, double dff);
	//	void UpdateDefault(); //call this functionn after change D or stn
	void Init(Model& model);
};
