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
 *
 */

#ifndef _ptens_Ptensors1bBatch
#define _ptens_Ptensors1bBatch

#include "diff_class.hpp"

#include "AtomsPackBatch.hpp"
#include "Ptensors1b.hpp"


namespace ptens{

  template<typename TYPE> class Ptensors1bBatch;
  template<typename TYPE> class Ptensors2bBatch;


  template<typename TYPE>
  class Ptensors1bBatch: public object_pack<Ptensors1b<TYPE> >, public cnine::diff_class<Ptensors1bBatch<TYPE> >{

    typedef object_pack<Ptennsors1b<TYPE> > BASE;

    using BASE::obj;
    using BASE::size;
    using BASE::operator[];
    using BASE::zip;
    using BASE::repr;
    using BASE::str;

    AtomsPackBatch atoms;


  public: // ----- Constructors ------------------------------------------------------------------------------


    //Ptensors1bBatch(){}

    Ptensors1bBatch(const AtomsPackBatch& _atoms):
      atoms(_atoms){}

    Ptensors1bBatch R(const AtomsPackBatch& a, const int _nc, const int _dev):
      Ptensors1bBatch(a){
      for(int i=0; i<atoms.size() i++)
	obj.push_back(Ptensors1b<TYPE>(atoms[i],_nc,_dev));
    }

    Ptensors1bBatch R(const AtomsPackBatch& a, const int _nc, const int fcode, const int _dev):
      Ptensors1bBatch(a){
      for(int i=0; i<atoms.size() i++)
	obj.push_back(Ptensors1b<TYPE>(atoms[i],_nc,fcode,_dev));
    }


  public: // ----- Spawning ----------------------------------------------------------------------------------


    Ptensors1bBatch copy() const{
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back((*this)[i].copy());
      return R;
    }

    Ptensors1b copy(const int _dev) const{
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back((*this)[i].copy(_dev));
      return R;
    }

    Ptensors1b zeros_like() const{
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back((*this)[i].zeros_like());
      return R;
    }

    Ptensors1b gaussian_like() const{
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back((*this)[i].gaussian_like());
      return R;
    }

    static Ptensors1b zeros_like(const Ptensors1b& x){
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back(Ptensors1b::zeros_like(x[i]));
      return R;
    }

    static Ptensors1b gaussian_like(const Ptensors1b& x){
      Ptensors1bBatch R(atoms);
      for(int i=0; i<size(); i++)
	R.obj.push_back(Ptensors1b::gaussian_like(x[i]));
      return R;
    }

    static Ptensors1b* new_zeros_like(const Ptensors1b& x){
      Ptensors1bBatch* R=new Ptensors1bBatch(atoms);
      for(int i=0; i<size(); i++)
	R->obj.push_back(Ptensors1b::zeros_like(x[i]));
      return R;
    }
    

  public: // ----- Access ------------------------------------------------------------------------------------


    static int getk(){
      return 0;
    }

    int get_dev() const{
      PTENS_ASSRT(size()>0);
      return (*this)[0].get_dev();
    }

    int get_nc() const{
      PTENS_ASSRT(size()>0);
      return (*this)[0].get_nc();
    }

    // save this or rebuild it each time?
    //AtomsPackBatch get_atoms(){
    //return atoms;
      //AtomsPackBatchObj* r=new AtomsPackBatchObj();
      //for(auto& p:obj)
      //r->obj.push_back(p.atoms.obj.atoms);
      //return r;
    //}


  public: // ---- Message passing ----------------------------------------------------------------------------


    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors1bBatch<TYPE> linmaps(const SOURCE& x){
      Ptensors1bBatch<float> R(x.atoms,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      R.add_linmaps(x);
      return R;
    }

    template<typename SOURCE, typename = typename std::enable_if<std::is_base_of<Ptensorsb<float>, SOURCE>::value, SOURCE>::type>
    static Ptensors1b<TYPE> gather(const SOURCE& x, const AtomsPackBatch& a){
      Ptensors1bBatch<float> R(a,x.get_nc()*vector<int>({1,1,2})[x.getk()],x.get_dev());
      Ptensors1bBatch<TYPE> R(a);
      for(int i=0; i<x.size(); i++)
	R.obj.push_back(PtensorsOb<TYPE>::gather(x[i],a[i]));
      return R;
    }

    void add_linmaps(const Ptensors1bBatch<TYPE>& x){
      //add(x);
    }

    void add_linmaps(const Ptensors1bBatch<TYPE>& x){
      //add(x.reduce0());
    }

    void add_linmaps(const Ptensors2bBatch<TYPE>& x){
      //add(x.reduce0());
    }

    template<typename SOURCE>
    void add_gather(const SOURCE& x){
      (atoms.overlaps_mmap(x.atoms))(*this,x);
    }

    template<typename OUTPUT>
    void add_gather_back(const OUTPUT& x){
      //x.atoms.overlaps_mmap(atoms).inv()(*this,x);
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Ptensors1bBatch";
    }


  };

}

#endif 
