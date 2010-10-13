/* cdp2.cpp
 * @author Quanli Wang, quanli@stat.duke.edu
 */
// stuff for loading data and initial values

#include <math.h>
#include "stdafx.h"
#include "Model.h"
#include "stdlib.h"

#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns
#include "newmatap.h"                // need matrix applications
#include "newmatio.h"                // need matrix output routines
#include "MersenneTwister.h"
#include "specialfunctions2.h"
#include "cdpprior.h"
#include "cdpbase.h"
#include "cdpresult.h"
#include "cdp.h"


void CDP::LoadData(Model& model) {
  if (!LoadFileData(model.mstrDataFile,&mX, model.mnN,model.mnD)) {
    std::cout << "Loading X failed!" << endl;
    exit (1);
  }
}

void CDP::LoadInits(Model& model,CDPResult& result, MTRand& mt)
{
  if(model.loadM)
    {
      if(!LoadFileData(model.Mfile,result.m,result.J,result.D))
	{
	  std::cout << "Loading m failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadPhi)
    {
      if(!LoadFileData(model.Phifile,result.Phi,result.J,result.D*result.D))
	{
	  std::cout << "Loading Phi failed!" << endl;
	  exit(1);
	}
      LowerTriangularMatrix temp;
      SymmetricMatrix temp2;
      for(int j=0;j<result.J;j++)
	{
	  temp2 = result.Phi[j]*prior.nu;
	  temp = Cholesky(temp2);
	  result.Phi_log_det[j] = msf.logdet(temp);
	  result.Phi_T_i[j]= temp.t().i();
	}
    }
  if(model.loadQ)
    {
      if(!LoadFileData(model.Qfile,result.q,result.J))
	{
	  std::cout << "Loading q failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadqV)
    {
      if(!LoadFileData(model.qVfile,result.qV,result.J))
	{
	  std::cout << "Loading qV failed!" << endl;
	  exit(1);
	}
    }
  

  if(model.loadMu)
    {
      if(!LoadFileData(model.Mufile,result.mu,result.J*result.T,result.D))
	{
	  std::cout << "Loading Mu failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadSigma)
    {
      if(!LoadFileData(model.Sigmafile,result.Sigma,result.J*result.T,result.D*result.D))
	{
	  std::cout << "Loading Sigma failed!" << endl;
	  exit(1);
	}
      int index;
      LowerTriangularMatrix temp;
	  IdentityMatrix ident(result.D);
		double epsilon = 10e-2;
      for(int j=0;j<result.J;j++)
	for(int t=0;t<result.T;t++)
	  {
	    index = result.GetIndex(j,t);
		//cout << result.Sigma[index] << endl;
		int flag = 0;
		int ITERTRY = 10;
		  do{
			  try {
				  temp << Cholesky(result.Sigma[index]);
				  flag=0;
			  }
			  catch (NPDException) {
				  flag++;
				  result.Sigma[index] << result.Sigma[index] + ident * epsilon;
				  if (flag >= ITERTRY) {
					  std::cout << "Input Covariance Matrix for the "<< t<< "th component could not be made pos-def after 10 tries" << endl;
					  exit(1);
				  }
			  }
		  } while (flag > 0);
	    result.Sigma_log_det[index] = msf.logdet(temp);
	    result.L_i[index] = temp.i();
	  }
    }
  if(model.loadP)
    {
      if(!LoadFileData(model.Pfile,result.p,result.J,result.T))
	{
	  std::cout << "Loading p failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadpV)
    {
      if(!LoadFileData(model.pVfile,result.pV,result.J,result.T))
	{
	  std::cout << "Loading pV failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadEta){
		if(!LoadFileData(model.etaFile,result.eta,result.J,result.T))
		{
			std::cout << "Loading pV failed!" << endl;
			exit(1);
		}
    }

  if(model.loadW)
    {
      if(!LoadFileData(model.Wfile,result.W,result.N))
	{
	  std::cout << "Loading W failed!" << endl;
	  exit(1);
	}
    } 
  else {
	  if (prior.J > 1) { //change this later for two layer model
		  for (int i = 0; i < prior.N;i++) {
			RowVector row(prior.D);
			for (int j =0; j < prior.D; j++) {
				row[j] = mX[i][j];
			}
			result.W[i] = sampleW(row,result.q, result.p, result.mu, result.L_i, result.Sigma_log_det, mt);
		  }
	  }
   }


  if(model.loadK)
    {
      if(!LoadFileData(model.Kfile,result.K,result.N))
	{
	  std::cout << "Loading K failed!" << endl;
	  exit(1);
	}
    }
  else  {
	  if (prior.J >1) {	//change this later for two layer model
		for (int i = 0; i < prior.N;i++) {
			RowVector row(prior.D);
			for (int j =0; j < prior.D; j++) {
				row[j] = mX[i][j];
			}
	
			int index = result.W[i];
			result.K[i] = sampleK(row,result.p[index],result.mu,result.L_i,index,result.Sigma_log_det,mt);
		}
	  }
   }
	
  if(model.Zrelabel && model.mstralgorithm=="mcmc")
    {
		int curclass;
		result.refZ = new int[result.N];
	   if(!LoadFileData(model.Zfile,result.refZ,result.N))
	{
		std::cout << "Loading Reference Z failed!" << endl;
		exit(1);
	}
		result.refZobs = new int[result.T];
		for (int i = 0; i<result.N; i++) {
			curclass = result.refZ[i];
			result.refZobs[curclass] = result.refZobs[curclass]+1;
		}
	}
		
  if(model.loadAlpha)
    {
      if(!LoadFileData(model.Alphafile,result.alpha,result.J))
	{
	  std::cout << "Loading alpha failed!" << endl;
	  exit(1);
	}
    }
  if(model.loadAlpha0)
    {
      if(!LoadFileData(model.Alpha0file,result.alpha0))
	{
	  std::cout << "Loading alpha0 failed!" << endl;
	  exit(1);
	}
    }
}

// used for Sigma and Phi: each row is a symmetric matrix with dimension=sqrt(columns)
#if defined(CDP_TBB)
bool CDP::LoadFileData(string FileName, concurrent_vector<SymmetricMatrix>& A, int rows, int columns)
#else
bool CDP::LoadFileData(string FileName, vector<SymmetricMatrix>& A, int rows, int columns)
#endif
{
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  
  double dim = sqrt((double)columns);
  double dCurrent = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < dim; j++) {
      for(int k=0;k<dim;k++)
	{
	  if (!theFile.eof()) {
	    theFile >> dCurrent;
	    //if(j<=k)
	      A[i][j][k] = dCurrent;
	  } else {
	    std::cout << "Not enough numbers to be read" << endl;
	    return false;
	  }
	}
    }
  }
  theFile.close();
  return true;

}

// used for alpha0
bool CDP::LoadFileData(string FileName, double& A)
{
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  
  double dCurrent = 0;
  if (!theFile.eof()) {
    theFile >> dCurrent;
    A = dCurrent;
  } else {
    std::cout << "Not enough numbers to be read" << endl;
    return false;
  }
  theFile.close();
  return true;

}

// used for m, p, pV, mu
#if defined(CDP_TBB)
bool CDP::LoadFileData(string FileName, concurrent_vector<RowVector>& A, int rows, int columns)
#else
bool CDP::LoadFileData(string FileName, vector<RowVector>& A, int rows, int columns)
#endif

{
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }

  double dCurrent = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (!theFile.eof()) {
	theFile >> dCurrent;
	A[i][j] = dCurrent;
      } else {
	std::cout << "Not enough numbers to be read" << endl;
	return false;
      }
    }
  }
  theFile.close();
  return true;
}

// used for w,k
bool CDP::LoadFileData(string FileName, int* A, int columns)
{
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  
  double dCurrent = 0.0;
  for (int j = 0; j < columns; j++) {
    if (!theFile.eof()) {
      theFile >> dCurrent;
      A[j] = ((int)dCurrent)-1;
    } else {
      std::cout << "Not enough numbers to be read" << endl;
      return false;
    }
  }
  theFile.close();
  return true;
}

// used for q, qV, alpha
bool CDP::LoadFileData(string FileName, RowVector& A, int columns)
{
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  
  double dCurrent = 0;
  for (int j = 0; j < columns; j++) {
    if (!theFile.eof()) {
      theFile >> dCurrent;
      A[j] = dCurrent;
    } else {
      std::cout << "Not enough numbers to be read" << endl;
      return false;
    }
  }
  theFile.close();
  return true;

}

// used for X
bool CDP::LoadFileData(string FileName, double*** A, int rows, int columns) {
  int i;
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  *A = new double*[rows];
  for (i = 0; i < rows; i++) {
	  (*A)[i] = new double[columns];
  }
  double dCurrent = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (!theFile.eof()) {
	theFile >> dCurrent;
	(*A)[i][j] = dCurrent;
      } else {
	std::cout << "Not enough numbers to be read" << endl;
	return false;
      }
    }
  }
  theFile.close();
  return true;
}
bool CDP::LoadFileData(string FileName, Matrix& A, int rows, int columns) {
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout << "Failed to open the file " << FileName.c_str() << endl;
    return false;
  }
  A = Matrix(rows,columns);
  double dCurrent = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (!theFile.eof()) {
	theFile >> dCurrent;
	A[i][j] = dCurrent;
      } else {
	std::cout << "Not enough numbers to be read" << endl;
	return false;
      }
    }
  }
  theFile.close();
  return true;
}
