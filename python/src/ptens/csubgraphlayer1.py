#
# This file is part of ptens, a C++/CUDA library for permutation 
# equivariant message passing. 
#  
# Copyright (c) 2023, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with cnine in the file LICENSE.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in 
# original or modified form) must retain this copyright notice and 
# must be accompanied by a verbatim copy of the license. 
#
#

import torch

import ptens as p
import ptens_base as pb 

import ptens.cptensorlayer1 as cptensorlayer1


class csubgraphlayer1(p.csubgraphlayer,cptensorlayer1):


    def __new__(cls,G,S,atoms,M):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        R=super().__new__(csubgraphlayer1,M)
        R.atoms=atoms
        R.G=G
        R.S=S
        return R

    @classmethod
    def zeros(self,G,S,nvecs,nc,device='cpu'):
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        S.set_evecs()
        atoms=pb.csubgraphatoms_cache.subgraphs(G.obj,S.obj,nvecs)
        M=torch.zeros([len(atoms),nvecs,nc],device=device)
        return csubgraphlayer1(G,S,atoms,M)

    @classmethod
    def randn(self,G,S,nvecs,nc,device='cpu'):
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        S.set_evecs()
        atoms=pb.csubgraphatoms_cache.subgraphs(G.obj,S.obj,nvecs)
        M=torch.randn([len(atoms),nvecs,nc],device=device)
        return csubgraphlayer1(G,S,atoms,M)

    @classmethod
    def from_tensor(self,G,S,M):
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==3
        S.set_evecs()
        atoms=pb.csubgraphatoms_cache.subgraphs(G.obj,S.obj,M.size(1))
        return csubgraphlayer1(G,S,atoms,M)

    
    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        return csubgraphlayer1(x.G,x.S,x.atoms,super().linmaps(x))


    # ---- Gather ------------------------------------------------------------------------------------------


    @classmethod
    def gather(self,S,x):
        atoms=pb.csubgraphatoms_cache.subgraphs(x.G.obj,S.obj,x.get_nvecs())
        return csubgraphlayer1(x.G,x.S,atoms,super().gather(atoms,x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "csubgraphlayer1(len="+str(self.size(0))+",nvecs="+str(self.size(1))+",nc="+str(self.get_nc())+")"

    def __str__(self,indent=""):
        r=indent+"csubgraphlayer1:\n"
        r=r+self.backend().__str__(indent+"  ")
        return r




