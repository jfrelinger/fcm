#if defined(CDP_TBB)
class WSampler {							
CDP *my_cdp;
CDPResult *my_result;
public:									
	void operator() ( const blocked_range<size_t>& r ) const {	
		MTRand tmt;
		tmt.seed();
		RowVector row(my_cdp->prior.D);
		for ( size_t i = r.begin(); i != r.end(); ++i ) {		
			for (int j =0; j < my_cdp->prior.D; j++) {
				row[j] = my_cdp->mX[i][j];
			}
			  if(cdp->mcsamplew)
			    my_result->W[i] = my_cdp->sampleW(row,my_result->q,my_result->m,
				my_result->Phi_T_i,my_cdp->prior.nu,my_cdp->prior.gamma,my_result->Phi_log_det,tmt);
			int wi = my_result->W[i];
			  if(cdp->mcsamplek)
			    my_result->K[i] = my_cdp->sampleK(row,my_result->p[wi],
							      my_result->mu,my_result->Sigma_T_i,wi,my_result->Sigma_log_det,tmt);
		}										
	}	
	WSampler(CDP *cdp, CDPResult *result) : 	
	my_cdp(cdp), my_result(result) { }										
};	
/*
class KSampler {							
CDP *my_cdp;
CDPResult *my_result;
int *my_clusternum;
vector<int> *my_clusterlist;
public:									
	void operator() ( const blocked_range<size_t>& r ) const {	
		MTRand tmt;
		tmt.seed();
		RowVector row(my_cdp->mX.Ncols());
		for ( size_t i = r.begin(); i != r.end(); ++i ) {		
			int index = (*my_clusterlist)[i];
			for (int j =0; j < my_cdp->mX.Ncols(); j++) {
					row[j] = my_cdp->mX[index][j];
				}
				my_result->K[index] = my_cdp->sampleK(row,my_result->p[*my_clusternum],
					my_result->mu,my_result->Sigma_T_i,*my_clusternum,my_result->Sigma_log_det,tmt);

		}
	}	
	KSampler(CDP *cdp, CDPResult *result, int *clusternum, vector<int> *clusterlist) : 	
	my_cdp(cdp), my_result(result),my_clusternum(clusternum), my_clusterlist(clusterlist) { }										
};	
*/
class MuSigmaSampler {							
CDP *my_cdp;
CDPResult *my_result;
int *my_clusternum;
vector<vector<int> > *my_clusterlist;
public:									
	void operator() ( const blocked_range<size_t>& r ) const {	
		MTRand tmt;
		tmt.seed();

		RowVector PostMu;
		SymmetricMatrix PostSigma;
		UpperTriangularMatrix uti;
		double logdet;
		for ( size_t t = r.begin(); t != r.end(); ++t ) {	
			int index = my_result->GetIndex(*my_clusternum,t);
			int flag = 0;
			int ITERTRY = 10;
			do {
				try {
					my_cdp->sampleMuSigma((*my_clusterlist)[t],my_cdp->prior.nu,my_cdp->prior.gamma,
						my_result->m[*my_clusternum],my_result->Phi[*my_clusternum],PostMu,PostSigma,uti,logdet,tmt);
					flag = 0;
				} catch (NPDException) {
					flag++;
					if (flag >= ITERTRY) {
						std::cout << "SampleMuSigma failed due to singular matrix after 10 tries" << endl;
						exit(1);
					}
				}
			}	while (flag > 0); 

			if(my_cdp->mcsamplemu) {      
				my_result->mu[index] = PostMu;
			}
			if(my_cdp->mcsampleSigma) {
				my_result->Sigma[index] = PostSigma;
				my_result->Sigma_log_det[index] = logdet;
				my_result->Sigma_T_i[index] = uti;
			}
		}
	}	
	MuSigmaSampler(CDP *cdp, CDPResult *result, int *clusternum, vector<vector<int> > *clusterlist) : 	
	my_cdp(cdp), my_result(result),my_clusternum(clusternum), my_clusterlist(clusterlist) { }										
};	
class ClusterSampler {							
CDP *my_cdp;
CDPResult *my_result;
vector<vector<int> > *my_clusterlist;
public:									
	void operator() ( const blocked_range<size_t>& r ) const {	
		MTRand tmt;
		tmt.seed();
		for ( size_t i = r.begin(); i != r.end(); ++i ) {		
			my_cdp->clusterIterate(i,(*my_clusterlist)[i],*my_result,tmt);
		}										
	}	
	ClusterSampler(CDP *cdp, CDPResult *result, vector<vector<int> > *clusterlist) : 	
	my_cdp(cdp), my_result(result),my_clusterlist(clusterlist) { }										
};	
/*
class ClusterSampler {							
CDP *my_cdp;
CDPResult *my_result;
public:									
	void operator() ( const blocked_range<size_t>& r ) const {	
		MTRand tmt;
		tmt.seed();
		vector<vector<int> >w1d;
		my_cdp->partition(my_result->W,w1d);
		//std::cout << "MT iterate" << endl;
		for ( size_t i = r.begin(); i != r.end(); ++i ) {		
			my_cdp->clusterIterate(i,w1d[i],*my_result,tmt);
		}
		
	}	
	ClusterSampler(CDP *cdp, CDPResult *result) : 	
	my_cdp(cdp), my_result(result) { }										
};	
*/
#endif
