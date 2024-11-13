/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */

#ifndef _Ptens_GraphObj
#define _Ptens_GraphObj

#include "once.hpp"
#include "sparse_graph.hpp"
#include "FindPlantedSubgraphs.hpp"
//#include "FindPlantedSubgraphs2.hpp"
#include "SubgraphObj.hpp"
#include "AtomsPack.hpp"
#include "Ltensor.hpp"


namespace ptens{


  class GgraphObj: public cnine::sparse_graph<int,int,int>{
  public:

    typedef cnine::sparse_graph<int,int,int> BASE;

    mutable unordered_map<shared_ptr<SubgraphObj>,AtomsPack> subgraphpack_cache;

    using BASE::nedges;

    //shared_ptr<GPUbundle> bundle;

    ~GgraphObj(){
    }


  public: //  ---- Constructors -------------------------------------------------------------------------------


    GgraphObj(){}

    GgraphObj(const int n, const initializer_list<pair<int,int> >& list): 
      BASE(n,list){}

    template<typename TYPE>
    GgraphObj(const cnine::TensorView<TYPE>& M):
      BASE(M){}

    template<typename TYPE>
    GgraphObj(const cnine::TensorView<TYPE>& M, const cnine::TensorView<TYPE>& L):
      BASE(M){
      set_labels(L);
    }


    GgraphObj(const initializer_list<pair<int,int> >& list, const int n): 
      BASE(n,list){}

    GgraphObj(int n, const cnine::TensorView<int>& M): 
      GgraphObj(n){
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      for(int i=0; i<M.dims(1); i++)
	set(M(0,i),M(1,i),1.0);
      //original_edges=AtomsPack(M.transp());
    }


  public: //  ---- Named constructors -------------------------------------------------------------------------


    static GgraphObj random(const int _n, const float p=0.5){
      return BASE::random(_n,p);
    }

    static GgraphObj from_edges(int n, const cnine::TensorView<int>& M){
      GgraphObj R(n);
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      for(int i=0; i<M.dims(1); i++)
	R.set(M(0,i),M(1,i),1.0);
      //R.original_edges=AtomsPack(M.transp());
      return R;
    }

    static GgraphObj from_edges(const cnine::TensorView<int>& M){
      return GgraphObj::from_edges(M.max()+1,M);
    }

    static GgraphObj* from_edges_p(const cnine::TensorView<int>& M){
      auto R=new GgraphObj(M.max()+1);
      PTENS_ASSRT(M.ndims()==2);
      PTENS_ASSRT(M.dim(0)==2);
      for(int i=0; i<M.dims(1); i++)
	R->set(M(0,i),M(1,i),1.0);
      return R;
    }


//     GgraphObj(int n, const cnine::TensorView<int>& M):
//       GgraphObj(n){
//       PTENS_ASSRT(M.ndims()==2);
//       PTENS_ASSRT(M.dim(0)==2);
//       for(int i=0; i<M.dims(1); i++)
// 	set(M(0,i),M(1,i),1.0);
//       original_edges=AtomsPack(M.transp());
//     }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    GgraphObj(const BASE& x):
      BASE(x){
    }

    //GgraphObj(const GgraphObj& x):
    //BASE(x){
    //cout<<"GGraph copied"<<endl;
    //}


  public: // ---- Access --------------------------------------------------------------------------------------


    cnine::TensorView<int> edge_list() const{
      int N=nedges()*2;
      cnine::TensorView<int> R({2,N},0,0);
      int t=0;
        for_each_edge([&](const int i, const int j, const int v){
	  R.set(0,t,i);
	  R.set(1,t,j);
	  R.set(0,t+1,j);
	  R.set(1,t+1,i);
	  t+=2;
	});
      return R;
    }

    void set_labels(const cnine::TensorView<int>& L){
      labels.reset(L);
      labeled=true;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    GgraphObj permute(const cnine::permutation pi) const{
      cnine::TensorView<int> A({2,nedges()},0,0);
      int t=0;
      for_each_edge([&](const int i, const int j, const int v){
	  A.set(0,t,pi(i));
	  A.set(1,t,pi(j));
	  t++;
	});
      return BASE(getn(),A);
     }


    cnine::once<AtomsPack> edges=cnine::once<AtomsPack>([&](){
	auto R=new AtomsPackObj(nedges(),2,cnine::fill_raw());
	int i=0;
	for(auto& p:data)
	  for(auto& q:p.second)
	    if(p.first<q.first){
	      R->set(i,0,p.first);
	      R->set(i,1,q.first);
	      i++;
	    }
	return AtomsPack(R);
      });


  public: // ---- Subgraphs ----------------------------------------------------------------------------------


    AtomsPack subgraphs(const shared_ptr<SubgraphObj>& S) const{
      auto it=subgraphpack_cache.find(S);
      if(it!=subgraphpack_cache.end()) return it->second;
      else{
	cnine::flog timer("GgraphObj::finding subgraphs");

	if(S->getn()==1 && S->labeled==false && S->nedges()==0){
	  AtomsPack r(getn());
	  subgraphpack_cache[S]=r;
	  return r;
	}

	if(S->getn()==2 && S->labeled==false && S->nedges()==1){
	  AtomsPack r;
	  for_each_edge([&](const int i, const int j, const int v){
	      if(i<j) r.push_back({i,j});});
	  subgraphpack_cache[S]=r;
	  return r;
	}

	AtomsPack r(new AtomsPackObj(cnine::TensorView<int>(cnine::FindPlantedSubgraphs<int>(*this,*S))));
	subgraphpack_cache[S]=r;
	return r;
      }
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    static string classname(){
      return "ptens::GgraphObj";
    }

    string repr() const{
      return "<Ggraph N="+to_string(getn())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Ggraph on "<<to_string(getn())<<" vertices:"<<endl;
      oss<<dense().str(indent+"  ");
      if(labeled){
	oss<<indent<<"Labels:"<<endl;
	oss<<labels.str(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GgraphObj& x){
      stream<<x.str(); return stream;}


  };

}

#endif 

