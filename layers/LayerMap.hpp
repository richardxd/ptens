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

#ifndef _ptens_LayerMap
#define _ptens_LayerMap

#include "LayerMapObj.hpp"


namespace ptens{


  class LayerMap{
  public:
    
    shared_ptr<LayerMapObj> obj;

    LayerMap(const shared_ptr<LayerMapObj>& _obj):
      obj(_obj){}

    static LayerMap overlaps_map(const AtomsPack& out, const AtomsPack& in, const int min_overlaps=1){
      PTENS_ASSRT(min_overlaps==1);

      if(ptens_global::overlaps_maps_cache.contains(*out.obj,*in.obj))
	return (ptens_global::overlaps_maps_cache(*out.obj,*in.obj));

      auto r=LayerMapObj::overlaps_map(*out.obj,*in.obj,min_overlaps);
      ptens_global::overlaps_maps_cache.insert(*out.obj,*in.obj,r);
      return LayerMap(r);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "LayerMap";
    }

    string repr() const{
      return "<LayerMap>";
    }

    string str(const string indent="") const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const LayerMap& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
