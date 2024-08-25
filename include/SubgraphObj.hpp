/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Kondor Risi
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _ptens_SubgraphObj
#define _ptens_SubgraphObj

#include "Ptens_base.hpp"
#include "Tensor.hpp"
#include "SymmEigendecomposition.hpp"
#include "sparse_graph.hpp"
#include "Tensor.hpp"


namespace ptens{


  class SubgraphObj: public cnine::sparse_graph<int,int,int>{
  public:

    typedef cnine::sparse_graph<int,int,int> BASE;
    typedef cnine::TensorView<int> rtensor;

    using BASE::BASE;
    using BASE::operator==;
    using BASE::dense;

    mutable cnine::TensorView<float> evecs;
    mutable vector<int> eblocks;

    SubgraphObj(const SubgraphObj& x):
      BASE(x),
      evecs(x.evecs),
      eblocks(x.eblocks){
    }

  public: // ---- Constructors -------------------------------------------------------------------------------


    SubgraphObj(const cnine::TensorView<int>& M, const cnine::TensorView<int>& L):
      BASE(M){
      labels.reset(L);
      labeled=true;
    }

    SubgraphObj(const vector<pair<int,int> >& list): 
      SubgraphObj([](const vector<pair<int,int> >& list){
	  int t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t); return t+1;}(list)){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    SubgraphObj(const int _n, const initializer_list<pair<int,int> >& list): 
      SubgraphObj(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    SubgraphObj(const int _n, const initializer_list<pair<int,int> >& list, const cnine::TensorView<int>& _labels): 
      BASE(_n,list,_labels){}

    SubgraphObj(const int n, const cnine::Tensor<int>& M):
      BASE(n){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      for(int i=0; i<M.dims(1); i++)
	set(M(0,i),M(1,i),1.0);
    }

    SubgraphObj(const int n, const cnine::TensorView<int>& _edges, const cnine::TensorView<int>& _labels):
      SubgraphObj(n,_edges){
      labels.reset(_labels);
      labeled=true;
    }

    SubgraphObj(const int n, const cnine::TensorView<int>& _edges, const cnine::TensorView<float>& _evecs, const cnine::TensorView<float>& evals):
      SubgraphObj(n,_edges){
      evecs.reset(_evecs);
      make_eblocks(evals);
    }

    SubgraphObj(const int n, const cnine::TensorView<int>& _edges, const cnine::TensorView<int>& _labels, const cnine::TensorView<float>& _evecs, const cnine::TensorView<float>& evals):
      SubgraphObj(n,_edges,_labels){
      evecs.reset(_evecs);
      make_eblocks(evals);
    }


  public: 


    bool has_espaces() const{
      return eblocks.size()>0;
    }

    cnine::TensorView<float> get_evecs() const{
      make_eigenbasis();
      return evecs;
    }

    void make_eigenbasis() const{
      if(eblocks.size()>0) return;
      int n=getn();

      auto A=dense();
      cnine::TensorView<float> L=cnine::TensorView<float>::zero({n,n});
      A.for_each([&](const int i, const int j, const int& v){
	  L.set(i,j,v);});
      //L.view2().add(dense().view2()); 
      for(int i=0; i<n; i++){
	float t=0; 
	for(int j=0; j<n; j++) t+=L(i,j);
	L.inc(i,i,-t);
      }

      auto eigen=cnine::SymmEigendecomposition<float>(L);
      evecs.reset(eigen.U());
      make_eblocks(eigen.lambda());
    }

    void set_evecs(const cnine::TensorView<float>& _evecs, const cnine::TensorView<float>& _evals) const{
      const_cast<SubgraphObj&>(*this).evecs.reset(_evecs);
      const_cast<SubgraphObj&>(*this).make_eblocks(_evals);
    }

    void make_eblocks(const cnine::TensorView<float>& evals) const{
      PTENS_ASSRT(evals.dims.size()==1);
      PTENS_ASSRT(getn()==evals.dims[0]);
      eblocks.clear();
      for(int i=0; i<evals.dims[0];){
	float t=evals(i);
	int start=i;
	while(i<evals.dims[0] && std::abs(evals(i)-t)<10e-5) i++;
	eblocks.push_back(i-start);
      }
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "SubgraphObj";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Subgraph on "<<to_string(getn())<<" vertices:"<<endl;
      oss<<dense().str(indent+"  ");
      if(labeled){
	oss<<indent<<"Labels:"<<endl;
	oss<<labels.str(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SubgraphObj& x){
      stream<<x.str(); return stream;}

  };

}


namespace std{
  template<>
  struct hash<ptens::SubgraphObj>{
  public:
    size_t operator()(const ptens::SubgraphObj& x) const{
      return hash<cnine::sparse_graph<int,int,int> >()(x);
    }
  };
}


#endif 

