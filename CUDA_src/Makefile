CXX = g++

#C_DEBUG = -g

INSTALL = /usr/bin/install -SC

PYTHON = /usr/bin/python2.6
PYCONFIG = $(PYTHON)-config

CXXFLAGS = $(C_DEBUG) -fPIC -DPYWRAP `$(PYCONFIG) --cflags` -Dmain=oldmain
LDFLAGS = $(C_DEBUG) -fPIC -DPYWRAP  `$(PYCONFIG) --ldflags`
INCPATH = -I../src/statistics/cdp_ext/matrix -I../src/statistics/cdp_ext/random -I../src/statistics/cdp_ext/hoomd -I../src/statistics/cdp_ext/cdp `$(PYCONFIG) --includes`
LIBPATH = `$(PYCONFIG) --libs`

CUDA = 1

##### CUDA definitions

CUDA_PATH = /usr/local/cuda

OSNAME := $(shell uname)
ifeq ($(OSNAME),Linux)
	CUDA_SDK_PATH   := /usr/local/cuda_sdk/C
	BOOST_INC_PATH 	:= /usr/include/boost
	BOOST_LIB_PATH 	:= /usr/lib64/
	#BOOST_LIB	:= boost_thread
endif
ifeq ($(OSNAME),Darwin)
	CUDA_SDK_PATH   := "/Developer/GPU Computing/C"
	BOOST_INC_PATH 	:= /usr/local/include
	BOOST_LIB_PATH 	:= /usr/local/lib
	BOOST_LIB	:= boost_thread
endif


NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_DBG_FLAGS = -Xcompiler -fno-strict-aliasing,-fPIC
ifdef CUDA
	CXXFLAGS += -DCDP_CUDA
	CUDA_OBJS =     obj/EMCDPBaseCUDA.cpp_o \
			obj/CDPBaseCUDA.cpp_o \
			obj/EMmvNormalPDF_kernel.cu_o \
			obj/mvNormalPDF_kernel.cu_o \
			obj/sampleFromMeasure_kernel.cu_o \
			obj/EMCalcPi_kernel.cu_o \
			obj/calcSigma_kernel.cu_o \
			obj/calcSumPi_kernel.cu_o \
			obj/CUDASharedFunctions.cpp_o \
			obj/CUDASharedFunctions_kernel.cu_o \
			obj/shiftMeasure_kernel.cu_o  \
			obj/argMaxMeasure_kernel.cu_o

	CUDA_HEADERS = CUDA/CDPBaseCUDA.h \
			CUDA/EMCDPBaseCUDA.h \
			CUDA/EMCDPBaseCUDA_kernel.h \
			CUDA/CDPBaseCUDA_kernel.h \
			CUDA/CUDASharedFunctions.h 
	CUDA_INC = -I$(CUDA_PATH)/include -I$(BOOST_INC_PATH)
	CUDA_SDK_COMMONDIR = $(CUDA_SDK_PATH)/common
	CUDA_SDK_INC = -I$(CUDA_SDK_COMMONDIR)/inc
	CUDA_LIB  = -L$(CUDA_PATH)/lib64 -L$(CUDA_SDK_PATH)/lib -L$(CUDA_SDK_COMMONDIR)/lib -lcuda -lcudart -lcutil -lcublas \
		-L$(BOOST_LIB_PATH) 
	INCPATH += -ICUDA  $(CUDA_INC)

else
	CUDA_OBJS =
	CUDA_HEADERS=
	CUDA_INC =
	CUDA_SDK_COMMONDIR =
	CUDA_SDK_INC =
	CUDA_LIB  =
endif
			
### Python
PYTHONINC = `$(PYCONFIG) --includes` \
			`$(PYTHON) -c'import numpy;print "-I%s" % numpy.get_include()'`

cdp_obj = 	obj/cdp_em.o obj/cdp.o obj/cdp2.o obj/cdp4.o \
			obj/cdpbase.o obj/cdpprior.o obj/cdpresult.o \
			obj/cdpresultem.o obj/Model.o
rand_obj = obj/SpecialFunctions.o obj/specialfunctions2.o
matrix_obj = obj/bandmat.o \
			obj/cholesky.o \
			obj/hholder.o \
			obj/jacobi.o \
			obj/myexcept.o \
			obj/newmat1.o \
			obj/newmat2.o \
			obj/newmat3.o \
			obj/newmat4.o \
			obj/newmat5.o \
			obj/newmat6.o \
			obj/newmat7.o \
			obj/newmat8.o \
			obj/newmat9.o \
			obj/newmatex.o \
			obj/newmatrm.o \
			obj/sort.o \
			obj/submat.o 

munkres_obj = obj/Munkres.o

py_obj = obj/cdpcluster.o obj/mvnpdf.o obj/cdpcluster_wrap.o obj/cdpemcluster.o

OBJ = $(cdp_obj) $(rand_obj) $(matrix_obj) $(py_obj) $(CUDA_OBJS) $(munkres_obj)
### Python files
_cdp.so: $(py_obj) $(cdp_obj) $(CUDA_OBJS) $(rand_obj) $(matrix_obj) $(munkres_obj)
	$(CXX) $(LDFLAGS) -shared $(cdp_obj) $(matrix_obj) $(rand_obj) $(munkres_obj) $(CUDA_OBJS)  obj/cdpcluster_wrap.o  obj/cdpcluster.o obj/mvnpdf.o obj/cdpemcluster.o $(INCPATH) $(LIBPATH) $(PYTHONINC) -L. $(CUDA_LIB) -lstdc++ -o _cdp.so

obj/cdpcluster.o:	../src/statistics/cdp_ext/wrapper/cdpcluster.cpp ../src/statistics/cdp_ext/wrapper/cdpcluster.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

obj/cdpemcluster.o:	../src/statistics/cdp_ext/wrapper/cdpemcluster.cpp ../src/statistics/cdp_ext/wrapper/cdpemcluster.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

obj/mvnpdf.o: ../src/statistics/cdp_ext/wrapper/mvnpdf.cpp ../src/statistics/cdp_ext/wrapper/mvnpdf.h
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)


obj/cdpcluster_wrap.o: ../src/statistics/cdp_ext/wrapper/cdpcluster_wrap.cpp
	$(CXX) $(CXXFLAGS)  $(INCPATH) $(PYTHONINC) $(CUDA_INC) $(CUDA_SDK_INC) -c ../src/statistics/cdp_ext/wrapper/cdpcluster_wrap.cpp -o obj/cdpcluster_wrap.o

obj/Munkres.o:		../src/statistics/cdp_ext/munkres/Munkres.cpp ../src/statistics/cdp_ext/munkres/Munkres.h
		$(CXX) $(CXXFLAGS) -c ../src/statistics/cdp_ext/munkres/Munkres.cpp -o $@ $(INCPATH)

##### CDP files

obj/%.o:           	../src/statistics/cdp_ext/cdp/%.cpp
		$(CXX) $(CXXFLAGS) -c ../src/statistics/cdp_ext/cdp/$*.cpp -o $@ $(INCPATH) -I../src/statistics/cdp_ext/munkres/ $(CUDA_INC) $(CUDA_SDK_INC)


#cdps:       	$(cdp_obj) $(CUDA_OBJS)
#		$(CXX) $(CXXFLAGS) $(PROFILE)  -o cdps-cuda $(cdp_obj) $(CUDA_OBJS) -L. -lm -lstdc++ $(LIBPATH) -lmat -l rng $(CUDA_LIB)

#cdpp.o:		cdpp.cpp cdpp.h cdp.h cdpbase.h cdpresult.h cdpprior.h Model.h	

obj/cdp.o:		../src/statistics/cdp_ext/cdp/cdp.cpp ../src/statistics/cdp_ext/cdp/cdp.h ../src/statistics/cdp_ext/cdp/cdpbase.h ../src/statistics/cdp_ext/cdp/cdpresult.h ../src/statistics/cdp_ext/cdp/cdpprior.h  ../src/statistics/cdp_ext/cdp/Model.h
obj/cdp_em.o:	../src/statistics/cdp_ext/cdp/cdp_em.cpp ../src/statistics/cdp_ext/cdp/cdp_em.h ../src/statistics/cdp_ext/cdp/cdp.h ../src/statistics/cdp_ext/cdp/cdpbase.h ../src/statistics/cdp_ext/cdp/cdpresult.h ../src/statistics/cdp_ext/cdp/cdpresultem.h ../src/statistics/cdp_ext/cdp/cdpprior.h ../src/statistics/cdp_ext/cdp/Model.h

obj/cdp2.o:		../src/statistics/cdp_ext/cdp/cdp.h ../src/statistics/cdp_ext/cdp/cdp2.cpp ../src/statistics/cdp_ext/cdp/Model.h 

obj/cdp3.o:		../src/statistics/cdp_ext/cdp/cdp.h ../src/statistics/cdp_ext/cdp/cdp.cpp ../src/statistics/cdp_ext/cdp/cdp3.cpp ../src/statistics/cdp_ext/cdp/Model.h

obj/Model.o:	../src/statistics/cdp_ext/cdp/Model.cpp ../src/statistics/cdp_ext/cdp/Model.h

obj/cdpbase.o:	../src/statistics/cdp_ext/cdp/cdpbase.cpp ../src/statistics/cdp_ext/cdp/cdpbase.h

obj/cdpresult.o:	../src/statistics/cdp_ext/cdp/cdpresult.cpp ../src/statistics/cdp_ext/cdp/cdpresult.h

obj/cdpprior.o:	../src/statistics/cdp_ext/cdp/cdpprior.cpp ../src/statistics/cdp_ext/cdp/cdpprior.h

obj/cdpresultem.o:	../src/statistics/cdp_ext/cdp/cdpresultem.cpp ../src/statistics/cdp_ext/cdp/cdpresultem.h



#### CUDA files

obj/%.c_o : CUDA/%.c
	$(GCC) $(PROFILE) -c $< -o $@ $(LOCAL_INC)

obj/%.cpp_o : CUDA/%.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) $(LOCAL_INC) $(CUDA_INC) $(CUDA_SDK_INC) $(PROFILE) -c $< -o $@
	
obj/%.cu_o : CUDA/%.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCC_DBG_FLAGS) -c $< -o $@ -I. $(INCPATH) $(LOCAL_INC) $(CUDA_INC) $(CUDA_SDK_INC) -DUNIX

### Random
obj/%.o:           	../src/statistics/cdp_ext/random/%.cpp
		$(CXX) $(CXXFLAGS) -c ../src/statistics/cdp_ext/random/$*.cpp -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

### Matrix
obj/%.o:           	../src/statistics/cdp_ext/matrix/%.cpp
		$(CXX) $(CXXFLAGS) -c ../src/statistics/cdp_ext/matrix/$*.cpp -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

	
### clean
.PHONY : clean
clean :
	$(RM) $(OBJ)
	$(RM) _cdp.so


### install
install : _cdp.so
	$(INSTALL) _cdp.so `$(PYTHON) -c 'from fcm.statistics import _cdp; print _cdp.__file__'`
