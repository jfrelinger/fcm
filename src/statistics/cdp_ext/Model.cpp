// Model.cpp: implementation of the Model class.
//
//////////////////////////////////////////////////////////////////////

#include "Model.h"
#pragma warning(disable:4786)
#pragma warning(disable : 4996)

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Model::Model()
{
  mstrDataFile = "";
  mnN = 0;
  mnD = 2;
  mnJ = 10;
  mnT = 10;
  
  mdm0 = RowVector(mnD); mdm0=0;
  mdphi0 = 4;
  mdnu0 = 1;
  mdgamma = 0.1;
  mdlambda0 = 0.25;
  mdnu = 1;
  mde0 = 0.1;
  mdf0 = 0.1;
  mdee = 0.1;
  mdff = 0.1;
  
  mnBurnin = 100;
  mnIter = 1000;
  
  mnSeed = 1138;

  // by default sample everything
  samplem = true;
  samplePhi = true;
  samplew = true;
  sampleq = true;
  samplealpha0 = true;
  samplemu = true;
  sampleSigma = true;
  samplek = true;
  samplep = true;
  samplealpha = true;

  //  singleDP=false; // a flag for enabling the single layer Dp mixture of normals

  //by default don't load any parameter values
  loadW=false;
  loadMu=false;
  loadSigma=false;
  loadK=false;
  loadM=false;
  loadPhi=false;
  loadP=false;
  loadQ=false;
  loadpV=false;
  loadqV=false;
  loadAlpha=false;
  loadAlpha0=false;

  Wfile="";
  Mufile="";
  Sigmafile="";
  Kfile="";
  Mfile="";
  Phifile="";
  Pfile="";
  Qfile="";
  pVfile="";
  qVfile="";
  Alphafile="";
  Alpha0file="";
}

string Model::ToLower(string str)
{
	//new implementation for GNU
	char *newstr = strdup(str.c_str());
	int i = 0;
	while (newstr[i] != '\0') {
		newstr[i] = tolower(newstr[i]);
		i++;
	}
	return newstr;
}

int Model::ToInt(string str)
{
	return atoi(str.c_str());
}

double Model::ToDouble(string str)
{
	return atof(str.c_str());
}

string Model::ToString(int value) {
	char  buffer[10]; 
	sprintf(buffer, "%d", value);
	string result(buffer);
	return result;
	cout << result.c_str() << endl;
}


string Model::ToString(double value) {
	char  buffer[10]; 
	sprintf(buffer, "%4.1f", value);
	string result(buffer);
	return result;
}

string Model::trim(string s)
{
	if (s.empty()) return s;
	string ret;
	for (int i = 0; i < (int)s.length();  i++) {
		if (!isspace(s.at(i)) && !iscntrl(s.at(i)))
			ret.append(s.substr(i,1));
	}
	return ret;
}

// for extracting a row vector from whatever is after the = 
RowVector Model::ToRowVector(string str)
{
  //  cout << "START" << str << "END" << endl;
  int pos = 0;
  if ((pos = str.find("=")) != -1) {
    str = str.substr(pos + 1);  // get rid of the junk before and including the equal sign
  }
  //  cout << "START" <<  str << "END" << endl;

  // get rid of leading white spaces
  int i;
  for(i=0;isspace(str.at(i)) && i<(int)str.length();i++)
    ;
  str = str.substr(i);
  //  cout << "START" << str<< "END" << endl;

  // get rid of trailing white spaces
  for(i=(int)str.length()-1;isspace(str.at(i)) && i>=0;i--)
    ;
  str = str.substr(0,i+1);
  //  cout << "START" << str << "END"<<endl;

  // count white spaces in between numbers
  int nspace=0;
  bool wasspace=0;    
  for(i=0;i<(int)str.length();i++)
    {
      if(isspace(str.at(i)) && !wasspace)
	{
	  nspace++;wasspace=true;
	}
      else
	{
	  wasspace=false;
	}
    }
  //  cout << "nspace=" << nspace << endl;

  RowVector vec(nspace+1);vec=0;
  
  // pop off the substrings, converting them to doubles and putting them in the vector
  int startpos=0;
  int numlength=0;
  int vecpos=0;
  wasspace=false;
  
  for(i=0;i<(int) str.length();i++)
    {
      if(wasspace && !isspace(str.at(i))) // if this is the start of a new number
	{
	  startpos=i;wasspace=false;numlength=1;
	}
      else if(!wasspace && !isspace(str.at(i))) // if this is the middle of a new number
	{
	  numlength++;
	}
      if(!wasspace && (isspace(str.at(i)) || iscntrl(str.at(i)) || i==(int)str.length()-1)) // if this is the end of a new number
	{
	  wasspace=true;
	  //	  cout << "vecpos=" << vecpos << " startpos=" << startpos << " numlength="<< numlength << " string=\""<< str.substr(startpos,numlength) << "\"";
	  vec[vecpos++] = ToDouble(str.substr(startpos,numlength));
	  //	  cout << " vec=" << vec[vecpos-1] << endl;
	}
      // if this is the middle of some between number whitespace  do nothing
    }
  return vec;
}

bool Model::Load(string FileName){
  bool setm0 = 0;
  int BufferSize = 4096;
  char* theLine = new char[BufferSize];
  ifstream theFile(FileName.c_str());
  if (theFile.fail()) {
    std::cout <<  "Failed to open the model file!" << std::endl;
    return false;
  }
  
  int nLineCount = 0;
  while (!theFile.eof()) {
    theFile.getline(theLine, BufferSize);
    nLineCount++;
    string theline(theLine);
    string Name(""), Value("");
    theline = trim(theline);			//so space is not allowed, should be improved later
    if (theline.length() && (theline.c_str()[0] != '#'))
      {
	int pos = 0;
	if ((pos = theline.find("=")) != -1) {
	  Name = theline.substr(0, pos);
	  Value = theline.substr(pos + 1);
	}
	if (Name == "" && Value == "") {
	} else if (Name == "" || Value == "") {
	  cout << "Invalid Parameter!" << endl;
	  cout << theLine << endl;
	  return false;
	} else {
	  string name = ToLower(Name);
	  string value = ToLower(Value);
	  if (name == "datafile") {
	    mstrDataFile = Value;
	  } else if(name=="wfile"){
	    loadW=true;
	    Wfile=Value;
	  } else if(name=="mufile"){
	    loadMu=true;
	    Mufile=Value;
	  } else if(name=="sigmafile"){
	    loadSigma=true;
	    Sigmafile=Value;
	  } else if(name=="kfile"){
	    loadK=true;
	    Kfile=Value;
	  } else if(name=="mfile"){
	    loadM=true;
	    Mfile=Value;
	  } else if(name=="phifile"){
	    loadPhi=true;
	    Phifile=Value;	  
	  } else if(name=="pfile"){
	    loadP=true;
	    Pfile=Value;
	  } else if(name=="qfile"){
	    loadQ=true;
	    Qfile=Value;
	  } else if(name=="pvfile"){
	    loadpV=true;
	    pVfile=Value;
	  } else if(name=="qvfile"){
	    loadqV=true;
	    qVfile=Value;
	  } else if(name=="alphafile"){
	    loadAlpha=true;
	    Alphafile=Value;
	  } else if(name=="alpha0file"){
	    loadAlpha0=true;
	    Alpha0file=Value;
	  } else if (name == "n") {
	    mnN = ToInt(value);
	  } else if (name == "d") {
	    mnD = ToInt(value);
	    setm0 = 1;
	  } else if (name == "j") {
	    mnJ = ToInt(value);
	  } else if (name == "t") {
	    mnT = ToInt(value);
	  } else if (name == "burnin") {
	    mnBurnin = ToInt(value);
	  } else if (name == "iter") {
	    mnIter = ToInt(value);
	  } else if (name == "seed") {
	    mnSeed = ToInt(value);
	  } else if (name == "nu") {
	    mdnu = ToDouble(value);
	  } else if (name == "nu0") {
	    mdnu0 = ToDouble(value);
	  } else if (name == "lambda0") {
	    mdlambda0 = ToDouble(value);
	  } else if (name == "phi0") {
	    mdphi0 = ToDouble(value);
	  } else if (name == "gamma") {
	    mdgamma = ToDouble(value);
	  } else if (name == "m0") {
	    mdm0 = ToRowVector(theLine);
	    setm0=false;
	    if(mdm0.Ncols()!=mnD)
	      {
		cout << "m0 not of length D" << endl;
		cout << theLine << endl;
		return false;
	      }
	  } else if (name == "e0") {
	    mde0 = ToDouble(value);
	  } else if (name == "f0") {
	    mdf0 = ToDouble(value);
	  } else if (name == "ee") {
	    mdee = ToDouble(value);
	  } else if (name == "ff") {
	    mdff = ToDouble(value);
	  } else if (name=="samplem"){
	    samplem = (ToInt(value)==1);
	  } else if (name=="samplephi"){
	    samplePhi = (ToInt(value)==1);
	  } else if (name=="samplew"){
	    samplew = (ToInt(value)==1);
	  } else if (name=="sampleq"){
	    sampleq = (ToInt(value)==1);
	  } else if (name=="samplealpha0"){
	    samplealpha0 = (ToInt(value)==1);
	  } else if (name=="samplemu"){
	    samplemu = (ToInt(value)==1);
	  } else if (name=="samplesigma"){
	    sampleSigma = (ToInt(value)==1);
	  } else if (name=="samplek"){
	    samplek = (ToInt(value)==1);
	  } else if (name=="samplep"){
	    samplep = (ToInt(value)==1);
	  } else if (name=="samplealpha"){
	    samplealpha = (ToInt(value)==1);
// 	  } else if (name=="singleDP"){
// 	    singleDP = (ToInt(value)==1);
	  } else {
	    cout << "Unknown Parameter" << endl; //to be refined later
	    cout << theLine << endl;
	    return false;
	  }
	  
	}
      }
  }
  if(setm0){mdm0 = RowVector(mnD);mdm0=0;}
  delete[] theLine;
  if (mstrDataFile == "") {
    std::cout << endl << "Warning: Text file "  << FileName.c_str() << " might come from a different platform" << endl;
    std::cout << "Suggestion: Use dos2unix/unis2dos or similar tools to convert the file first" << endl;
    return false;
  }
  //	Print();
  return true;
}

bool Model::Print(){
	cout << "## Current Settings ##" << endl << endl;

	cout << "#data section" << endl;
	cout << "N = " << mnN << endl;
	cout << "D = " << mnD << endl;
	if (mstrDataFile != "") {
		cout << "datafile = " << mstrDataFile.c_str() << endl;
	} else {
		cout << "#datafile = " << endl;
	}	
	cout << endl;

	cout << "#prior section" << endl;
	cout << "J = " << mnJ << endl;
	cout << "T = " << mnT << endl;
	cout << "m0 = ";
	for(int i=0;i<mnD;i++)
	  cout << mdm0[i] << " ";
	cout << endl;
	cout << "nu0 = " << mdnu0 << endl;
	cout << "gamma = " << mdgamma << endl;
	cout << "phi0 = " << mdphi0 << endl;
	cout << "lambda0 = " << mdlambda0 << endl;
	cout << "e0 = " << mde0 << endl;
	cout << "f0 = " << mdf0 << endl;
	cout << "nu = " << mdnu << endl;
	cout << "ee = " << mdee << endl;
	cout << "ff = " << mdff << endl;
	cout << endl;
	
	cout << "#MCMC section" << endl;
	cout << "burnin = " << mnBurnin << endl;
	cout << "iter = " << mnIter << endl;
	cout << "seed = " << mnSeed << endl<<endl;

	cout << "#Turn on/off individual sampling steps"<<endl;
	cout << "samplem = " << samplem << endl;
	cout << "samplePhi = " << samplePhi << endl;
	cout << "samplew = " << samplew << endl;
	cout << "sampleq = " << sampleq << endl;
	cout << "samplealpha0 = " << samplealpha0 << endl;
	cout << "samplemu = " << samplemu << endl;
	cout << "sampleSigma = " << sampleSigma << endl;
	cout << "samplek = " << samplek << endl;
	cout << "samplep = " << samplep << endl;
	cout << "samplealpha = " << samplealpha << endl << endl;
	
// 	cout << "# Enable single layer DP mixture model" << endl;
// 	cout << "singleDP = " << singleDP << endl << endl;
	
	cout << "#Load initial MCMC values from files"<<endl;
	if (loadAlpha0){
	  cout << "Alpha0file = " << Alpha0file.c_str() << endl;
	} else {
	  cout << "#Alpha0file = " << endl;
	}	

	if (loadM){
	  cout << "Mfile = " << Mfile.c_str() << endl;
	} else {
	  cout << "#Mfile = " << endl;
	}
	
	if (loadPhi){
	  cout << "Phifile = " << Phifile.c_str() << endl;
	} else {
	  cout << "#Phifile = " << endl;
	}	
	if (loadW){
	  cout << "Wfile = " << Wfile.c_str() << endl;
	} else {
	  cout << "#Wfile = " << endl;
	}	
	if (loadQ){
	  cout << "Qfile = " << Qfile.c_str() << endl;
	} else {
	  cout << "#Qfile = " << endl;
	}	
	if (loadqV){
	  cout << "qVfile = " << qVfile.c_str() << endl;
	} else {
	  cout << "#qVfile = " << endl;
	}	
	if (loadAlpha){
	  cout << "Alphafile = " << Alphafile.c_str() << endl;
	} else {
	  cout << "#Alphafile = " << endl;
	}	
	if (loadMu){
	  cout << "Mufile = " << Mufile.c_str() << endl;
	} else {
	  cout << "#Mufile = " << endl;
	}	
	if (loadSigma){
	  cout << "Sigmafile = " << Sigmafile.c_str() << endl;
	} else {
	  cout << "#Sigmafile = " << endl;
	}	
	if (loadK){
	  cout << "Kfile = " << Kfile.c_str() << endl;
	} else {
	  cout << "#Kfile = " << endl;
	}	
	if (loadP){
	  cout << "Pfile = " << Pfile.c_str() << endl;
	} else {
	  cout << "#Pfile = " << endl;
	}	
	if (loadpV){
	  cout << "pVfile = " << pVfile.c_str() << endl;
	} else {
	  cout << "#pVfile = " << endl;
	}	
	
	
       	cout << endl;

	cout.flush();
	return true;
}

bool Model::Save(string FileName){
	ofstream theFile(FileName.c_str());
	if (theFile.fail()) {
		cout << "Failed to create file!" << endl;
		return false;
	}
	theFile << "#Version 1.0" << endl << endl;

	theFile << "#data section" << endl;
	theFile << "N = " << mnN << endl;
	theFile << "D = " << mnD << endl;
	if (mstrDataFile != "") {
		theFile << "DataFile = " << mstrDataFile.c_str() << endl;
	} else {
		theFile << "#DataFile = " << endl;
	}	
	theFile << endl;

	theFile << "#prior section" << endl;
	theFile << "J = " << mnJ << endl;
	theFile << "T = " << mnT << endl;
	theFile << "m0 = ";
	for(int i=0;i<mnD;i++)
	  theFile << mdm0[i] << " ";
	theFile << endl;
	theFile << "nu0 = " << mdnu0 << endl;
	theFile << "gamma = " << mdgamma << endl;
	theFile << "phi0 = " << mdphi0 << endl;
	theFile << "lambda0 = " << mdlambda0 << endl;
	theFile << "e0 = " << mde0 << endl;
	theFile << "f0 = " << mdf0 << endl;
	theFile << "nu = " << mdnu << endl;
	theFile << "ee = " << mdee << endl;
	theFile << "ff = " << mdff << endl;
	theFile << endl;
	
	theFile << "#MCMC section" << endl;
	theFile << "burnin = " << mnBurnin << endl;
	theFile << "iter = " << mnIter << endl;
	theFile << "seed = " << mnSeed << endl << endl;

	theFile << "#Turn on/off individual sampling steps"<<endl;
	theFile << "samplem = " << samplem << endl;
	theFile << "samplePhi = " << samplePhi << endl;
	theFile << "samplew = " << samplew << endl;
	theFile << "sampleq = " << sampleq << endl;
	theFile << "samplealpha0 = " << samplealpha0 << endl;
	theFile << "samplemu = " << samplemu << endl;
	theFile << "sampleSigma = " << sampleSigma << endl;
	theFile << "samplek = " << samplek << endl;
	theFile << "samplep = " << samplep << endl;
	theFile << "samplealpha = " << samplealpha << endl << endl;
	
// 	theFile << "# Enable single layer DP mixture model" << endl;
// 	theFile << "singleDP = " << singleDP << endl << endl;
	
	theFile << "#Load initial MCMC values from files"<<endl;
	if (loadAlpha0){
	  theFile << "Alpha0file = " << Alpha0file.c_str() << endl;
	} else {
	  theFile << "#Alpha0file = " << endl;
	}	

	if (loadM){
	  theFile << "Mfile = " << Mfile.c_str() << endl;
	} else {
	  theFile << "#Mfile = " << endl;
	}
	
	if (loadPhi){
	  theFile << "Phifile = " << Phifile.c_str() << endl;
	} else {
	  theFile << "#Phifile = " << endl;
	}	
	if (loadW){
	  theFile << "Wfile = " << Wfile.c_str() << endl;
	} else {
	  theFile << "#Wfile = " << endl;
	}	
	if (loadQ){
	  theFile << "Qfile = " << Qfile.c_str() << endl;
	} else {
	  theFile << "#Qfile = " << endl;
	}	
	if (loadqV){
	  theFile << "qVfile = " << qVfile.c_str() << endl;
	} else {
	  theFile << "#qVfile = " << endl;
	}	
	if (loadAlpha){
	  theFile << "Alphafile = " << Alphafile.c_str() << endl;
	} else {
	  theFile << "#Alphafile = " << endl;
	}	
	if (loadMu){
	  theFile << "Mufile = " << Mufile.c_str() << endl;
	} else {
	  theFile << "#Mufile = " << endl;
	}	
	if (loadSigma){
	  theFile << "Sigmafile = " << Sigmafile.c_str() << endl;
	} else {
	  theFile << "#Sigmafile = " << endl;
	}	
	if (loadK){
	  theFile << "Kfile = " << Kfile.c_str() << endl;
	} else {
	  theFile << "#Kfile = " << endl;
	}	
	if (loadP){
	  theFile << "Pfile = " << Pfile.c_str() << endl;
	} else {
	  theFile << "#Pfile = " << endl;
	}	
	if (loadpV){
	  theFile << "pVfile = " << pVfile.c_str() << endl;
	} else {
	  theFile << "#pVfile = " << endl;
	}	
	
	theFile << endl;

	theFile.flush();
	theFile.close();
	return true;
}

