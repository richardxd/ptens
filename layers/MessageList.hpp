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

#ifndef _ptens_MessageList
#define _ptens_MessageList

#include "AtomsPackObj.hpp"
#include "MessageListObj.hpp"


namespace ptens{

  class MessageList{
  public:

    shared_ptr<MessageListObj> obj;


    static MessageList overlap(const AtomsPackObj& x, const AtomsPackObj& y):
      obj(new MessageListObj(x,y)){}

    pair<const cnine::hlists<int>&, const cnine::hlists<int>&> lists() const{
      return pair<const cnine::hlists<int>&, const cnine::hlists<int>&>(obj->in,obj->out);
    }


  };

}

#endif 
