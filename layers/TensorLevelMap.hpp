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

#ifndef _ptens_TensorLevelMap
#define _ptens_TensorLevelMap

#include "TensorLevelMapObj.hpp"


namespace ptens{

  class AtomsPackObj;


  class TensorLevelMap{
  public:
    
    shared_ptr<TensorLevelMapObj> obj;

    TensorLevelMap(){
      PTENS_ASSRT(false);}

    TensorLevelMap(const shared_ptr<TensorLevelMapObj>& x):
      obj(x){}

    TensorLevelMap(TensorLevelMapObj* x):
      obj(x){}

    //TensorLevelMap(const AtomsPack& _in_atoms, const AtomsPack& _out_atoms):
    //obj(new TensorLevelMapObj(*_in_atoms.obj,*_out_atoms.obj)){}


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      return obj->is_empty();
    }

    bool is_graded() const{
      return obj->is_graded();
    }

    const AtomsPack atoms() const{
      return obj->atoms;
    }

    const AindexPack& in() const{
      return *obj->in;
    }

    const AindexPack& out() const{
      return *obj->out;
    }

    std::shared_ptr<cnine::GatherMap> get_bmap() const{
      return obj->get_bmap();
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TransferrMap";
    }

    string repr() const{
      return "TensorLevelMap";
    }

    string str(const string indent="") const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const TensorLevelMap& v){
      stream<<v.str(); return stream;}


  };

}

#endif 
