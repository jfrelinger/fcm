#pragma once
#if defined(CDP_MPI)
#include "cdp.h"
class CDPP :
	public CDP
{
public:
	CDPP(void);
	~CDPP(void);

	void InitPBuffer(void);
	bool piterate(CDPResult& result);

	//W routines
	int GetWBufferSize();
	void PrepareWBuffer(CDPResult& result);
	void PostWBuffer(CDPResult& result);
	int SampleW(int index,CDPResult& result);
	void BroardcastW(CDPResult& result);
	void SampleAllW(CDPResult& result);

	//K routines
	int GetKBufferSize();
	void PrepareKBuffer(CDPResult& result);
	void PostKBuffer(CDPResult& result);
	int SampleK(int index, int wi, CDPResult& result);
	void BroardcastK(CDPResult& result);
	void SampleAllK(CDPResult& result);
	
	//mu sigma routines
	void SampleMuSigma(int index, int j, CDPResult& result);
	void SampleAllMuSigma(CDPResult& result);
	int GetMuSigmaBufferSize();
	void PackUnpackMuSigmaBuffer(CDPResult& result, double* workresult, int flag);

public:
	double* wbuffer;
	double* kbuffer;
	vector<list<int> >wk2d;
	vector<list<int> >w1d;
	vector<vector<int> > KJ;
};
#endif
