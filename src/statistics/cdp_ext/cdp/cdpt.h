/* cdpt.h
 * @author Quanli Wang, quanli@stat.duke.edu
 */
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

			if (my_cdp->mcsamplew && my_cdp->mcsamplek) {
				my_cdp->sampleWK(row, my_result->q, my_result->p, my_result->mu, my_result->L_i,
				my_result->Sigma_log_det, my_result->W[i], my_result->K[i], tmt);
			} else if (my_cdp->mcsamplew && !my_cdp->mcsamplek) {
			    my_result->W[i] = my_cdp->sampleW(row,my_result->q, my_result->p, my_result->mu,
				my_result->L_i, my_result->Sigma_log_det,tmt);
			} else if (!my_cdp->mcsamplew && my_cdp->mcsamplek) {
				int wi = my_result->W[i];
				if(my_cdp->mcsamplek) {
					my_result->K[i] = my_cdp->sampleK(row,my_result->p[wi],
									  my_result->mu,my_result->L_i,wi,my_result->Sigma_log_det,tmt);
				}
			}
		}										
	}	
	WSampler(CDP *cdp, CDPResult *result) : 	
	my_cdp(cdp), my_result(result) { }										
};	
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
		LowerTriangularMatrix li;
		double logdet;
		for ( size_t t = r.begin(); t != r.end(); ++t ) {	
			int index = my_result->GetIndex(*my_clusternum,(int)t);
			int flag = 0;
			int ITERTRY = 10;
			do {
				try {
					my_cdp->sampleMuSigma((*my_clusterlist)[t],my_cdp->prior.nu,my_cdp->prior.gamma,
						my_result->m[*my_clusternum],my_result->Phi[*my_clusternum],PostMu,PostSigma,li,logdet,tmt);
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
				my_result->L_i[index] = li;
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
			my_cdp->clusterIterate_one(*my_result,tmt);
		}										
	}	
	ClusterSampler(CDP *cdp, CDPResult *result, vector<vector<int> > *clusterlist) : 	
	my_cdp(cdp), my_result(result),my_clusterlist(clusterlist) { }										
};	
#endif
