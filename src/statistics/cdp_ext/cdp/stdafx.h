#if !defined(propa_stdafx)
#define propa_stdafx
#pragma warning(disable:4786)
#pragma warning(disable:4146)
#pragma warning(disable:4996)
#pragma warning(disable:4800)
//#define CDP_TBB
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <algorithm>
#include <ctime>
#include <set>
#include <stack>

#define WIN32_LEAN_AND_MEAN
#if defined(CDP_TBB)
	#include "tbb/task_scheduler_init.h"
	#include "tbb/parallel_for.h"
	#include "tbb/blocked_range.h"
	#include "tbb/concurrent_vector.h"
	using namespace tbb;
#endif
using namespace std;
//#define MULTI_GPU 
//#define CDP_MEANCOV
#endif 
