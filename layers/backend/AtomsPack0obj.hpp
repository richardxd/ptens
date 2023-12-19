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

#ifndef _ptens_AtomsPack0obj
#define _ptens_AtomsPack0obj

#include "map_of_lists.hpp"
#include "AtomsPackObj.hpp"
#include "GatherMapProgram.hpp"


namespace ptens{

  template<typename DUMMY>
  class AtomsPack0obj{
  public:

    typedef cnine::ptr_indexed_object_bank<AtomsPack0obj<DUMMY>,GatherMapProgram> TBANK0;
    typedef cnine::ptr_indexed_object_bank<AtomsPack1obj<DUMMY>,GatherMapProgram> TBANK1;
    typedef cnine::ptr_indexed_object_bank<AtomsPack2obj<DUMMY>,GatherMapProgram> TBANK2;


    shared_ptr<AtomsPackObj> atoms;


  public: // ---- Constructors ------------------------------------------------------------------------------


    AtomsPackOb1(const shared_ptr<AtomsPackObj>& _atoms):
      atoms(_atoms){}


  public: // ---- Access ------------------------------------------------------------------------------------


    int size() const{
      return atoms->size();
    }

    int index_of(const int i){
      return i;
    }


  public: // ---- Transfer maps -----------------------------------------------------------------------------


    GatherMapProgram overlaps_map(const AtomsPack0obj<DUMMY>& x){
      return overlaps_map0(x);}

    GatherMapProgram overlaps_map(const AtomsPack1obj<DUMMY>& x){
      return overlaps_map1(x);}

    GatherMapProgram overlaps_map(const AtomsPack2obj<DUMMY>& x){
      return overlaps_map2(x);}

    
    // 0 <- 0
    TBANK0 overlaps_map0=TBANK0([&](const AtomsPack0obj<DUMMY>& y){
	auto[in,out]=atoms->overlaps_mlist(*y.atoms).lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in.size(); m++){
	  int in_tensor=in.head(m);
	  int out_tensor=out.head(m);
	  direct.push_back(index_of(out_tensor),y.index_of(in_tensor);});
      
	return GatherMapProgram(new GatherMapB(direct));
      });
  

    // 0 <- 1
    TBANK1 overlaps_map1=TBANK1([&](const AtomsPack0obj<DUMMY>& y){
      auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();

      map_of_lists2<int,int> direct;
      for(int m=0; m<in_list.size(); m++){
	int in_tensor=in_lists.head(m);
	int out_tensor=out_lists.head(m);
	direct.push_back(index_of(out_tensor),y.index_of(in_tensor,in_lists(m,0)));
      }

      return GatherMapProgram(new GatherMapB(direct));
    });


    // 0 <- 2
    TBANK2 overlaps_map2=TBANK2([&](const AtomsPack0obj<DUMMY>& y){
	auto[in_lists,out_lists]=atoms->overlaps_mlist(*y.atoms).lists();

	map_of_lists2<int,int> direct;
	for(int m=0; m<in_lists.size(); m++){
	  int in_tensor=in_lists.head(m);
	  int out_tensor=out_lists.head(m);
	  direct.push_back(index_of(out_tensor),y.index_of(in_tensor,in_lists(m,0),in_lists(m,0)));
	}

	return GatherMapProgram(new GatherMapB(direct));
      });



  };

}

#endif 
