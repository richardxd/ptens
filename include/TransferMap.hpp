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
 *
 */

#ifndef _ptens_TransferMap
#define _ptens_TransferMap

#include "SparseRmatrix.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "AindexPack.hpp"


namespace ptens{

  class TransferMap: public cnine::SparseRmatrix{
  public:
    
    typedef cnine::SparseRmatrix SparseRmatrix;
    using SparseRmatrix::SparseRmatrix;


  public: // ---- Construct from overlaps ------------------------------------------------------------------------------


    TransferMap(const cnine::Tensor<int>& y, const cnine::Tensor<int>& x):
      TransferMap(x.dim(1),y.dim(1)){
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(y.ndims()==2);
      const int kx=x.dims[1];
      const int ky=y.dims[1];

      for(int i=0; i<x.dims[0]; i++){
	for(int j=0; j<y.dims[0]; j++){

	  bool found=false;
	  for(int a=0; !found && a<kx; a++){
	    int t=x(i,a);
	    for(int b=0; !found && b<ky; b++)
	      if(y(j,b)==t) found=true;
	  }
	  if(found) set(i,j,1);

	}
      }
    }


    TransferMap(const cnine::Tensor<int>& y, const cnine::array_pool<int>& x):
      TransferMap(x.size(),y.dims[0]){
      CNINE_ASSRT(y.ndims()==2);
      const int ky=y.dims[1];

      for(int i=0; i<x.size(); i++){
	auto v=x(i);
	for(int j=0; j<y.dims[0]; j++){
	  
	  bool found=false;
	  for(int a=0; !found && a<v.size(); a++){
	    int t=v[a];
	    for(int b=0; !found && b<ky; b++)
	      if(y(j,b)==t) found=true;
	  }
	  if(found) set(i,j,1);
	  
	}
      }
    }

      
    TransferMap(const cnine::array_pool<int>& y, const cnine::Tensor<int>& x):
      TransferMap(x.dims[0],y.size()){
      CNINE_ASSRT(x.ndims()==2);
      const int kx=x.dims[1];

	for(int i=0; i<x.dims[0]; i++){
	  for(int j=0; j<y.size(); j++){
	    auto v=y(j);
	    
	    bool found=false;
	    for(int a=0; !found && a<kx; a++){
	      int t=x(i,a);
	      for(int b=0; !found && b<v.size(); b++)
		if(v[b]==t) found=true;
	    }
	    if(found) set(i,j,1);
	  }
	}
    }


    TransferMap(const cnine::array_pool<int>& y, const cnine::array_pool<int>& x):
      TransferMap(x.size(),y.size()){
      if(y.size()<10){
	for(int i=0; i<x.size(); i++){
	  auto v=x(i);
	  for(int j=0; j<y.size(); j++){
	    auto w=y(j);
	    if([&](){for(auto p:v) if(std::find(w.begin(),w.end(),p)!=w.end()) return true; return false;}())
	      set(i,j,1.0);
	  }
	}
      }else{
	unordered_map<int,vector<int> > map;
	for(int j=0; j<y.size(); j++){
	  auto w=y(j);
	  for(auto p:w){
	    auto it=map.find(p);
	    if(it==map.end()) map[p]=vector<int>({j});
	    else it->second.push_back(j);
	  }
	}
	for(int i=0; i<x.size(); i++){
	  auto v=x(i);
	  for(auto p:v){
	    auto it=map.find(p);
	    if(it!=map.end())
	      for(auto q:it->second)
		set(i,q,1.0);
	  }
	}
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //int getn() const{
    //return n;
    //}

    bool is_empty() const{
      for(auto q:lists)
	if(q.second->size()>0)
	  return false;
      return true;
    }

    void forall_edges(std::function<void(const int, const int, const float)> lambda, const bool self=0) const{
      for(auto& p: lists){
	int i=p.first;
	if(self) lambda(i,i,1.0);
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Intersects --------------------------------------------------------------------------------------------


    pair<AindexPack,AindexPack> intersects(const AtomsPack& inputs, const AtomsPack& outputs, const bool self=0) const{
      //cout<<n<<" "<<m<<" "<<inputs.size()<<" "<<outputs.size()<<endl;
      PTENS_ASSRT(outputs.size()==n);
      PTENS_ASSRT(inputs.size()==m);
      AindexPack in_indices;
      AindexPack out_indices;
      forall_edges([&](const int i, const int j, const float v){
	  Atoms in=inputs[j];
	  Atoms out=outputs[i];
	  Atoms common=out.intersect(in);
	  //in_indices.push_back(j,in(common));
	  //out_indices.push_back(i,out(common));
	  auto _in=in(common);
	  auto _out=out(common);
	  in_indices.push_back(j,_in);
	  out_indices.push_back(i,_out);
	  in_indices.count1+=_in.size();
	  in_indices.count2+=_in.size()*_in.size();
	  out_indices.count1+=_out.size();
	  out_indices.count2+=_out.size()*_out.size();
	    
	}, self);
      //out_indices.bmap=new cnine::GatherMap(get_bmap());
      //if(!bmap) bmap=std::shared_ptr<cnine::GatherMap>(new cnine::GatherMap(broadcast_map())); 
      //out_indices.bmap=bmap; //new cnine::GatherMap(get_bmap());
      return make_pair(in_indices, out_indices);
    }


  };

}

#endif 