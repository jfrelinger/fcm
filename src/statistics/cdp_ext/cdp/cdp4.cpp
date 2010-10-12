/*
 *  cdp4.cpp
 *  
 *
 *  Created by Andrew Cron on 7/14/10.
 *
 */

#include <iostream>
#include <iomanip>
#include <math.h>
#include "stdafx.h"
#include "Model.h"
#include "newmatap.h"
#include "newmatio.h"
#include "cdpresult.h"
#include "MersenneTwister.h"
#include "Munkres.h"
#include "specialfunctions2.h"
#include "cdpprior.h"
#include "cdpbase.h"
#include "cdp.h"


void CDP::ComponentRelabel(CDPResult& result)
{
	

	int T = result.T;
	int N = result.N;
	SquareMatrix dist(T);
	int refClass, curClass;
	int* relabeling = new int[T];
	
	Munkres m;

	
	//Find relabeling cost matrix:
	for (int i=0; i<T; i++) {
		curClass = result.refZobs[i];
		for (int j=0; j<T; j++) {
			dist.element(i,j) = (double)curClass;
		}
	}
	for (int i=0; i<N; i++) {
		refClass = result.refZ[i];
		curClass = result.Z[i];
		dist.element(refClass,curClass) = dist.element(refClass,curClass) - (double)1.0;
	}
	
	
	// For debugging
	/*
	std::cout << "COST" << endl;
	for ( int row = 0 ; row < T ; row++ ) {
		for ( int col = 0 ; col < T ; col++ ) {
			std::cout.width(2);
			std::cout << dist(row,col) << ",";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
	
	dist = m.solve(dist);
	
	// For debugging
	
	/*std::cout << "SOLUTION" << endl;
	for ( int row = 0 ; row < T ; row++ ) {
		for ( int col = 0 ; col < T ; col++ ) {
			std::cout.width(2);
			std::cout << dist(row,col) << ",";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/
	
	//Find relabeling pattern:
	for (int row = 0; row < T; row++) {
		for (int col = 0; col < T; col++) {
			if (dist.element(row,col)==1) {
				relabeling[row] = col;
			}
		}
	}
	
	// For Debugging
	/*
	std::cout << "RELABELING" << endl;
	for (int row = 0; row < T; row++){
		std::cout.width(2);
		std::cout << relabeling[row] << ",";
	}
	std::cout << endl;
	*/
	//Relabel the observations without using too much memory
	vector<RowVector> mu2(result.mu);
	for (int i = 0; i<T; i++) {result.mu[i] = mu2[relabeling[i]];}
	mu2.clear();
	
	vector<SymmetricMatrix> Sigma2(result.Sigma);
	for (int i = 0; i<T; i++) {result.Sigma[i] = Sigma2[relabeling[i]];}
	Sigma2.clear();
	
	vector<RowVector> p2(result.p);
	for (int i = 0; i<T; i++) {result.p[0][i] = p2[0][relabeling[i]];}
	p2.clear();
	
	vector<RowVector> pV2(result.pV);
	for (int i = 0; i<T; i++) {result.pV[0][i] = pV2[0][relabeling[i]];}
	pV2.clear();
	
	vector<RowVector> eta2(result.eta);
	for (int i = 0; i<T; i++) {result.eta[0][i] = eta2[0][relabeling[i]];}
	eta2.clear();
	
	vector<LowerTriangularMatrix> L_i2(result.L_i);
	for (int i = 0; i<T; i++) {result.L_i[i] = L_i2[relabeling[i]];}
	L_i2.clear();
	
	vector<double> Sigma_log_det2(result.Sigma_log_det);
	for (int i = 0; i<T; i++) {result.Sigma_log_det[i] = Sigma_log_det2[relabeling[i]];}
	Sigma_log_det2.clear();


}
