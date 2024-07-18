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

#ifndef _ptens_AindexPackB
#define _ptens_AindexPackB

#include <map>

#include "hlists.hpp"
#include "monitored.hpp"
#include "Atoms.hpp"
#include "GatherMapB.hpp"
#include "Ltensor.hpp"


namespace ptens{


  class AindexPackB: public cnine::Ltensor<int>{
  public:

    typedef cnine::Ltensor<int> TENSOR;

    int _max_nix=0;
    int count1=0;
    int count2=0;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AindexPackB(const int n, const int maxn):
      TENSOR({n,maxn+4}){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    /*
    AindexPack(const AindexPack& x):
      BASE(x){
      _max_nix=x._max_nix;
      count1=x.count1;
      count2=x.count2;
    }

    AindexPack(AindexPack&& x):
      BASE(std::move(x)){
      _max_nix=x._max_nix;
      count1=x.count1;
      count2=x.count2;
    }

    AindexPack& operator=(const AindexPack& x)=delete;
    */


  public: // ---- Access -------------------------------------------------------------------------------------

    
    int size() const{
      return TENSOR::dim(0);
    }

    int toffset(const int i) const{
      return (*this)(i,0);
    }

    int nix(const int i) const{
      return (*this)(i,1);
    }

    int soffset(const int i) const{
      return (*this)(i,2);
    }

    int ssize(const int i) const{
      return (*this)(i,3);
    }

    int ix(const int i, const int j) const{
      return (*this)(i,j+4);
    }

    vector<int> ix(const int i) const{
      int n=nix(i);
      vector<int> R(n);
      for(int j=0; j<n; j++)
	R[j]=(*this)(i,j);
      return R;
    }

    void set(const int i, const int _toffset, const int _nix, const int _soffset, const int _ssize, const vector<int> v){
      PTENS_ASSRT(i<dim(0));
      PTENS_ASSRT(v.size()<=dim(1)-4);
      TENSOR::set(i,0,_toffset);
      TENSOR::set(i,1,_nix);
      TENSOR::set(i,2,_soffset);
      TENSOR::set(i,3,_ssize);
      for(int j=0; j<v.size(); j++)
	TENSOR::set(i,j+4,v[j]);
    }

    //const cnine::GatherMap& get_bmap() const{
    //assert(bmap);
    //return *bmap;
    //}

    /*
    int* get_barr(const int _dev=0) const{
      assert(bmap);
      bmap->to_device(_dev);
      if(_dev==0) return bmap->arr;
      return bmap->arrg;
    }
    */

    
  public: // ---- Operations ---------------------------------------------------------------------------------

  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "AindexPackB";
    }
    
    string repr() const{
      return "<AindexPack[N="+std::to_string(size())+"]>";
    }

    friend ostream& operator<<(ostream& stream, const AindexPackB& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
 
