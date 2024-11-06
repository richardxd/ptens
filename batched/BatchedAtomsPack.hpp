/*
 * This file is part of ptens, a C++/CUDA library for permutation 
 * equivariant message passing. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 */


#ifndef _ptens_BatchedAtomsPack
#define _ptens_BatchedAtomsPack

#include "AtomsPack.hpp"
#include "BatchedAtomsPackObj.hpp"


namespace ptens{


  class BatchedAtomsPackBase{
  public:


    shared_ptr<BatchedAtomsPackObj> obj;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedAtomsPackBase():
      obj(new BatchedAtomsPackObj()){}

    BatchedAtomsPackBase(BatchedAtomsPackObj* _obj):
      obj(_obj){}

    //BatchedAtomsPackBase(BatchedAtomsPackObj&& _obj):
    //obj(new BatchedAtomsPackObj(_obj)){}

    BatchedAtomsPackBase(shared_ptr<BatchedAtomsPackObj> _obj):
      obj(_obj){}

    BatchedAtomsPackBase(const vector<AtomsPack>& x):
      obj(new BatchedAtomsPackObj()){
      for(auto& p:x)
	obj->push_back(p.obj);
    }

    //BatchedAtomsPackBase(const vector<AtomsPack>& x):
    //obj(new BatchedAtomsPackObj()){
    //for(auto& p:x)
    //obj->push_back(p.obj);
    //}


    //BatchedAtomsPackBase(const vector<AtomsPack>& x):
    //BatchedAtomsPackBase(new BatchedAtomsPackObj(cnine::mapcar<AtomsPack,shared_ptr<AtomsPackObj> >
    //	  (x,[](const AtomsPack& y){return y.obj;}))){}

    BatchedAtomsPackBase(const vector<vector<vector<int> > >& x):
      obj(new BatchedAtomsPackObj(x)){}

    BatchedAtomsPackBase(const initializer_list<initializer_list<initializer_list<int> > >& x):
      obj(new BatchedAtomsPackObj(x)){}



  public: // ----- Access ------------------------------------------------------------------------------------


    int size() const{
      return obj->size();
    }

    AtomsPack operator[](const int i) const{
      PTENS_ASSRT(i<size());
      return (*obj)(i);
    }

    vector<vector<vector<int> > > as_vecs() const{
      return obj->as_vecs();
    }

    bool operator==(const BatchedAtomsPackBase& x) const{
      if(obj.get()==x.obj.get()) return true;
      return (*obj)==(*x.obj);
    }


  public: // ---- Layout -------------------------------------------------------------------------------------


    int nrows0() const{return obj->nrows0();}
    int nrows0(const int i) const{return obj->nrows0(i);}
    int offset0(const int i) const{return obj->offset0(i);}

    int nrows1() const{return obj->nrows1();}
    int nrows1(const int i) const{return obj->nrows1(i);}
    int offset1(const int i) const{return obj->offset1(i);}

    int nrows2() const{return obj->nrows2();}
    int nrows2(const int i) const{return obj->nrows2(i);}
    int offset2(const int i) const{return obj->offset2(i);}


  public: // ---- Operations ---------------------------------------------------------------------------------


    //BatchedAtomsPack permute(const cnine::permutation& pi){
    //return BatchedAtomsPack(new BatchedAtomsPackObj(obj->permute(pi)));
    //} 
    
    //MessageListBatch overlaps_mlist(const BatchedAtomsPack& y){
    //return obj->overlaps_mlist(*y.obj);
    //}


    static BatchedAtomsPackBase cat(const vector<BatchedAtomsPackBase >& v){
      PTENS_ASSRT(v.size()>0);
      int N=v[0].size();
      auto R=new BatchedAtomsPackObj();
      for(int i=0; i<N; i++){
	vector<shared_ptr<AtomsPackObj> > w;
	for(auto& p: v)
	  w.push_back((*p.obj)(i));
	R->push_back(AtomsPackObj::cat(w));
      }
      return BatchedAtomsPackBase(R);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "BatchedAtomsPackBase";
    }

    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const BatchedAtomsPackBase& v){
      stream<<v.str(); return stream;}

  };


  template<int k>
  class BatchedAtomsPack: public BatchedAtomsPackBase{
  public:

    typedef BatchedAtomsPackBase BASE;

    using BASE::BASE;

    BatchedAtomsPack(const BatchedAtomsPackBase& x):
      BASE(x){}

  };

}

#endif 
