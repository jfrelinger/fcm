#include <math.h>
#include "stdafx.h"
#include "Model.h"


#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
//#if defined(CDP_TBB)
//	#include "tbb/concurrent_vector.h"
//	using namespace tbb;
//#endif
#include "cdpresult.h"
CDPResult::CDPResult(){
};
CDPResult::CDPResult(int nclusters, int ncomponents, int npoints, int dimension)
{
	J = nclusters;
	T = ncomponents;
	N = npoints;
	D = dimension;
	m.reserve(J);
	Phi.reserve(J);
	alpha = RowVector(J);

	mu.reserve(J * T);
	Sigma.reserve( J * T);
	p.reserve( J );
	pV.reserve(J );

	xmbar.reserve(N);
	xmubar.reserve(N);
	nmcits = 0;

	q = RowVector(J);
	qV = RowVector(J);
	
	W = new int[N];
	K = new int[N];

	postmufile.open("postmu.txt");
	if(postmufile.fail())
	  {
	    std::cout << "Failed to create postmu.txt" << endl;
	    exit(1);
	  }
	postpfile.open("postp.txt");
	if(postpfile.fail())
	  {
	    std::cout << "Failed to create postp.txt" << endl;
	    exit(1);
	  }

	postSigmafile.open("postSigma.txt");
	if(postSigmafile.fail())
	  {
	    std::cout << "Failed to create postSigma.txt" << endl;
	    exit(1);
	  }

	postqfile.open("postq.txt");
	if(postqfile.fail())
	  {
	    std::cout << "Failed to create postq.txt" << endl;
	    exit(1);
	  }

	postmfile.open("postm.txt");
	if(postmfile.fail())
	  {
	    std::cout << "Failed to create postm.txt" << endl;
	    exit(1);
	  }

	postPhifile.open("postPhi.txt");
	if(postPhifile.fail())
	  {
	    std::cout << "Failed to create postPhi.txt" << endl;
	    exit(1);
	  }


}

CDPResult::~CDPResult(void)
{
	delete [] W;
	delete [] K;
	postmufile.close();
	postpfile.close();
	postSigmafile.close();
	postmfile.close();
	postqfile.close();
	postPhifile.close();
	
}

/*************************************************************************
 * Functions for logging the predictive density during each step of mcmc *
 ************************************************************************/
bool CDPResult::SaveDraws(){

  SavePDraw();
  SaveMuDraw();
  SavePhiDraw();

  SaveQDraw();
  SaveMDraw();
  SaveSigmaDraw();
  return true;
}

bool CDPResult::SavePDraw(){
  for(int j=0;j<J;j++)
    {
      for(int t=0;t<T;t++)
	{
	  postpfile << q[j]*p[j][t] << "\t";
	}
      postpfile<<endl;
    }
  return true;
}

bool CDPResult::SaveMuDraw(){
  for (int i = 0; i < J*T; i++) {
    for (int j = 0; j < D; j++) {
      postmufile << mu[i][j] << "\t";
    }
    postmufile << endl;
  }
  return true;
}

bool CDPResult::SaveSigmaDraw(){
  for (int i = 0; i < J*T; i++) {
    for (int j = 0; j < D; j++) {
      for (int k = 0; k <= j; k++) {
	postSigmafile << Sigma[i][j][k] << "\t";
      }
      for (int k = j+1; k < D; k++) {
	postSigmafile << Sigma[i][k][j] << "\t";
      }
    }
    postSigmafile << endl;
  }
  return true;
}

bool CDPResult::SaveQDraw(){
  for(int j=0;j<J;j++)
    {
      postqfile << q[j] << "\t";
    }
  postqfile << endl;
  return true;
}

bool CDPResult::SaveMDraw(){
  for (int i = 0; i < J; i++) {
    for (int j = 0; j < D; j++) {
      postmfile << m[i][j] << "\t";
    }
    postmfile << endl;
  }
  return true;
}

bool CDPResult::SavePhiDraw()
{
  for (int i = 0; i < J; i++) {
    for (int j = 0; j < D; j++) {
      for (int k = 0; k <= j; k++) {
	postPhifile << Phi[i][j][k] << "\t";
      }
      for (int k = j+1; k < D; k++) {
	postPhifile << Phi[i][k][j] << "\t";
      }
    }
    postPhifile << endl;
  }
  return true;
}

/*********************************************************************
 * functions for saving all parameter values needed to continue mcmc
 ********************************************************************/
bool CDPResult::SaveFinal() {
	SaveK("lastk.txt");
	SaveW("lastw.txt");
	SaveMu("lastmu.txt");
	SaveSigma("lastSigma.txt");
	SaveM("lastm.txt");
	SavePhi("lastPhi.txt");
	SaveP("lastp.txt");
	SaveQ("lastq.txt");
	SaveAlpha("lastalpha.txt");
	SaveAlpha0("lastalpha0.txt");
	SavepV("lastpV.txt");
	SaveqV("lastqV.txt");
	return true;
}
bool CDPResult::SaveAlpha0(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
  theFile << alpha0 << endl;
  theFile.close();
  return true;
}

bool CDPResult::SaveAlpha(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
  for(int j=0;j<J;j++)
    {
      theFile << alpha[j] << "\t";
    }
  theFile << endl;
  theFile.close();
  return true;
}

bool CDPResult::SaveW(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	for (int i = 0; i < N; i++) {
	  theFile << W[i] +1<< endl;

	}
	//theFile << endl;
	theFile.close();
	return true;
}

bool CDPResult::SaveM(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < D; j++) {
			theFile << m[i][j] << "\t";
		}
		theFile << endl;
	}
	theFile.close();
	return true;
}

bool CDPResult::SavePhi(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < J; i++) {
		for (int j = 0; j < D; j++) {
			for (int k = 0; k <= j; k++) {
				theFile << Phi[i][j][k] << "\t";
			}
			for (int k = j+1; k < D; k++) {
				theFile << Phi[i][k][j] << "\t";
			}
		}
		theFile << endl;
	}
	theFile.close();
	return true;
}

bool CDPResult::SaveQ(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
  for(int j=0;j<J;j++)
    {
      theFile << q[j] << "\t";
    }
  theFile << endl;
  theFile.close();
  return true;
}

bool CDPResult::SaveqV(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
  for(int j=0;j<J;j++)
    {
      theFile << qV[j] << "\t";
    }
  theFile << endl;
  theFile.close();
  return true;
}

bool CDPResult::SaveK(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	for (int i = 0; i < N; i++) {
	  theFile << K[i]+1 << endl;
	}
	//theFile << endl;
	theFile.close();
	return true;
}


bool CDPResult::SaveMu(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < J*T; i++) {
		for (int j = 0; j < D; j++) {
			theFile << mu[i][j] << "\t";
		}
		theFile << endl;
	}
	theFile.close();
	return true;
}

bool CDPResult::SaveSigma(string FileName) {
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < J*T; i++) {
		for (int j = 0; j < D; j++) {
			for (int k = 0; k <= j; k++) {
				std::cout << i << j << k << std::endl;
				theFile << Sigma[i][j][k] << "\t";
			}
			for (int k = j+1; k < D; k++) {
				std::cout << i << j << k << std::endl;
				theFile << Sigma[i][k][j] << "\t";
			}
		}
		theFile << endl;
	}
	theFile.close();
	return true;
}

bool CDPResult::SaveP(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
  }
  //  int blah=1;
  for(int j=0;j<J;j++)
    {
      for(int t=0;t<T;t++)
	{
	  theFile << p[j][t] << "\t";
	}
      theFile<<endl;
    }
  theFile.close();
  return true;
}

bool CDPResult::SavepV(string FileName) {
  ofstream theFile(FileName.c_str());
  if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
  }
  //  int blah=1;
  for(int j=0;j<J;j++)
    {
      for(int t=0;t<T;t++)
	{
	  theFile << pV[j][t] << "\t";
	}
      theFile<<endl;
    }
  theFile.close();
  return true;
}


/*******************************************************
 * functions for maintaining and logging running means
 *******************************************************/
bool CDPResult::SaveBar() {
	SaveXMbar("postxmbar.txt");
	SaveXMubar("postxmubar.txt");
	return true;
}

bool CDPResult::SaveXMbar(string FileName)
{
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			theFile << xmbar[i][j]/nmcits << "\t";
		}
		theFile << endl;
	}
	theFile.close();
	return true;

}
bool CDPResult::SaveXMubar(string FileName)
{
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		std::cout << "Failed to create file " << FileName.c_str()  << endl;
		exit(1);
	}
	//	int D = mu[0].Ncols();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < D; j++) {
			theFile << xmubar[i][j]/nmcits << "\t";
		}
		theFile << endl;
	}
	theFile.close();
	return true;

}

void CDPResult::UpdateMeans()
{
  int j;
  RowVector foo(D);foo=0;
  SymmetricMatrix foo2(D);foo2=0;
  RowVector foo3(T);foo3=0;
  // if this is the first update, zero out the parameters
  if(nmcits==0)
    {
      for(j=0;j<N;j++)
	{
	  xmbar.push_back(foo);
	  xmubar.push_back(foo);
	}
    }
  
  for(j=0;j<N;j++)
    {
      xmbar[j]+=m[W[j]];
      xmubar[j]+=mu[GetIndex(W[j],K[j])];
    }
  nmcits++;
}
