/* Model.h
 * @author Quanli Wang, quanli@stat.duke.edu
 */
// Model.h: interface for the Model class.
//
//////////////////////////////////////////////////////////////////////
#include <map>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "newmatio.h"                // need matrix output routines
#include "newmatap.h"                // need matrix applications

using namespace std;

class Model  
{
public:
	Model();
	bool Load(string FileName);
	bool Save(string FileName);
	bool Print();
public:
	string mstralgorithm;
	string mstrDataFile;
	
	int mnN;
	int mnD;
	int mnJ;
	int mnT;
	
	RowVector mdm0;
	double mdlambda0;
	double mdphi0;
	double mdnu0;
	double mdgamma;
	double mdnu;
	double mde0;
	double mdf0;
	double mdee;
	double mdff;
	double mdaa;

	int mnCompSwaps;
	int mnBurnin;
	int mnIter;
	
	int mnPrintout;
	double mnErrorPerTol;		  // EM: Percent Difference stopping rule for diffence in posterior.

	int mnSeed;

	RowVector ToRowVector(string str);
	static double ToDouble(string str);
	static int ToInt(string str);
	static string ToLower(string str);
	static string ToString(int value);
	static string ToString(double value);
	static string trim(string s);

	// turn off/off individual sampling steps
	bool samplem;
	bool samplePhi;
	bool samplew;
	bool sampleq;
	bool samplealpha0;
	
	bool samplemu;
	bool sampleSigma;
	bool samplek;
	bool samplep;
	bool samplealpha;
	
	bool sampleEta;

//#if defined(CDP_CUDA)
	int startDevice;
	int numberDevices;
//#endif


	//	bool singleDP;

	// load sets of parameters
	bool loadW;
	bool loadMu;
	bool loadSigma;
	bool loadK;
	bool loadM;
	bool loadPhi;
	bool loadP;
	bool loadQ;
	bool loadpV;
	bool loadqV;
	bool loadAlpha;
	bool loadAlpha0;
	bool loadEta;

	int mnGPU_Sample_Chunk_Size;
	
	string Wfile;
	string Mufile;
	string Sigmafile;
	string Kfile;
	string Mfile;
	string Phifile;
	string Pfile;
	string Qfile;
	string pVfile;
	string qVfile;
	string Alphafile;
	string Alpha0file;
	string etaFile;

};

