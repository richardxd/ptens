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

#ifndef _ptens_AtomsPack0
#define _ptens_AtomsPack0

#include "AtomsPack.hpp"

namespace ptens{

  class AtomsPack0{
  public:


    shared_ptr<AtomsPackObj0> obj;


  public: // ---- Maps ---------------------------------------------------------------------------------------
    

    template<typename ATOMSPACKN>
    CompoundTransferMap transfer_map(const ATOMSPACKN& x){
      return CompoundTransferMap(obj->transfer_map(*x.obj));
    }


  };

}

#endif 
