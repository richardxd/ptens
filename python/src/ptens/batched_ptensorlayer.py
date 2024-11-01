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
import ptens_base as pb 

class batched_ptensorlayer(torch.Tensor):

    covariant_functions=[torch.Tensor.to,torch.Tensor.add,torch.Tensor.sub,torch.relu, torch.Tensor.mul, torch.nn.functional.linear, torch.nn.functional.batch_norm]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in batched_ptensorlayer.covariant_functions:
            r= super().__torch_function__(func, types, args, kwargs)
            for arg in args:
                if isinstance(arg, batched_ptensorlayer):
                    r.atoms=arg.atoms
                    r.G = arg.G
                    r.S = arg.S
                    break
        else:
            r= super().__torch_function__(func, types, args, kwargs)
            if isinstance(r,torch.Tensor):
                r=torch.Tensor(r)
        return r


    # ---- Operations ----------------------------------------------------------------------------------------


#    def __add__(self,y):
#        assert self.size()==y.size()
#        assert self.atoms==y.atoms
#        return self.from_matrix(self.atoms,super().__add__(y))


def matmul(x,y):
    return x.from_matrix(x.atoms,torch.matmul(x,y))

