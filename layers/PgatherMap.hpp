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

#ifndef _ptens_PgatherMap
#define _ptens_PgatherMap

#include "PgatherMapObj.hpp"
#include "AtomsPack.hpp"


namespace ptens{

  //class AtomsPackObj;


  class PgatherMap{
  public:
    
    shared_ptr<PgatherMapObj> obj;

    PgatherMap(){
      PTENS_ASSRT(false);}

    PgatherMap(const shared_ptr<PgatherMapObj>& x):
      obj(x){}

    const AindexPackB& in() const{
      return *obj->in_map;
    }

    const AindexPackB& out() const{
      return *obj->out_map;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "PgatherMap";
    }

    string repr() const{
      return "PgatherMap";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const PgatherMap& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
