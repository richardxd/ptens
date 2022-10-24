import torch

import ptens_base 
from ptens_base import ptensors1 as _ptensors1

import ptens.ptensors0 
import ptens.ptensors2 


class ptensors1(torch.Tensor):

    @classmethod
    def from_matrix(self,T,atoms):
        return Ptensors1_fromMxFn.apply(T,atoms)

    @classmethod
    def dummy(self):
        R=ptensors1(1)
        R.obj=_ptensors1.dummy()
        return R

    @classmethod
    def raw(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.raw(_atoms,_nc,_dev)
        return R

    @classmethod
    def zeros(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.zero(_atoms,_nc,_dev)
        return R

    @classmethod
    def randn(self, _atoms, _nc, _sigma=1.0, _device=0):
        R=ptensors1(1)
        R.obj=_ptensors1.gaussian(_atoms,_nc,_sigma,ptens.device_id(_device))
        return R

    @classmethod
    def sequential(self, _atoms, _nc, _dev=0):
        R=ptensors1(1)
        R.obj=_ptensors1.sequential(_atoms,_nc,_dev)
        return R


    # ----- Access -------------------------------------------------------------------------------------------


    def _get_grad(self):
        return self.obj.get_grad()
    
    def _view_of_grad(self):
        return self.obj.view_of_grad()
    
    def get_grad(self):
        R=ptensors1(1)
        R.obj=self.obj.get_grad()
        return R
    
    def view_of_grad(self):
        R=ptensors1(1)
        R.obj=self.obj.view_of_grad()
        return R


    def get_dev(self):
        return self.obj.get_dev()

    def get_nc(self):
        return self.obj.get_nc()

    def get_atoms(self):
        return self.obj.get_atoms()
    
    def atoms_of(self, i):
        return self.obj.atoms_of(i)

    def __getitem__(self,i):
        return Ptensors1_getFn.apply(self,i)
    
    def push_back(self, x):
        return self.obj.push_back(x)

    def randn_like(self,sigma=1.0):
        return ptensors1.randn(self.get_atoms(),self.get_nc(),sigma,self.get_dev())

    def torch(self):
        return Ptensors1_toMxFn.apply(self)

    def to(self, _device='cpu'):
        self.obj.to_device(ptens.device_id(_device))


    # ---- Operations ----------------------------------------------------------------------------------------

    
    def __add__(self,y):
        return Ptensors1_addFn.apply(self,y)

    def __mul__(self,y):
        return Ptensors1_mprodFn.apply(self,y)

    def linear(self,y,b):
        return Ptensors1_linearFn.apply(self,y,b)

    def concat(self,y):
        return Ptensors1_concatFn.apply(self,y)
    
    def relu(self,alpha=0.5):
        return Ptensors1_ReLUFn.apply(self,alpha)
        
    def inp(self,y):
        return Ptensors1_inpFn.apply(self,y)
    
    def diff2(self,y):
        return Ptensors1_diff2Fn.apply(self,y)


    def linmaps0(self):
        return Ptensors1_Linmaps0Fn.apply(self);

    def linmaps1(self):
        return Ptensors1_Linmaps1Fn.apply(self);

    def linmaps2(self):
        return Ptensors1_Linmaps2Fn.apply(self);


    def transfer0(self,_atoms,G):
        return Ptensors1_Transfer0Fn.apply(self,_atoms,G)

    def transfer1(self,_atoms,G):
        return Ptensors1_Transfer1Fn.apply(self,_atoms,G)

    def transfer2(self,_atoms,G):
        return Ptensors1_Transfer2Fn.apply(self,_atoms,G)


    def unite1(self,G):
        return Ptensors1_Unite1Fn.apply(self,G)
    
    def unite2(self,G):
        return Ptensors1_Unite2Fn.apply(self,G)

    
    def outer(self,y):
        if isinstance(y,ptens.ptensors0):
            return Ptensors1_Outer0Fn.apply(self,y)
        if isinstance(y,ptens.ptensors1):
            return Ptensors1_Outer1Fn.apply(self,y)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return self.obj.__repr__()

    def __str__(self):
        return self.obj.__str__()


# ------------------------------------------------------------------------------------------------------------


class Ptensors1_fromMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms):
        R=ptensors1(1)
        R.obj=_ptensors1(x,atoms)
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        return ctx.r.get_grad().torch(), None


class Ptensors1_toMxFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x.obj
        return x.obj.torch()
 
    @staticmethod
    def backward(ctx,g):
       R=ptensors1(1)
       ctx.x.add_to_grad(_ptensors1(g,ctx.x.get_atoms()))
       return R
    

class Ptensors1_getFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,i):
        R=ptens.ptensor1(x.obj[i].torch())
        R.atoms=x.atoms_of(i)
        ctx.x=x.obj
        ctx.i=i
        return R

    @staticmethod
    def backward(ctx,g):
        R=ptensors1(1)
        ctx.x.add_to_grad(ctx.i,g)
        return R,None

    
class Ptensors1_addFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptensors1(1)
        R.obj=_ptensors1(x.obj)
        R.obj.add(y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.r.get_gradp())
        ctx.y.add_to_grad(ctx.r.get_gradp())
        return ptensors1(1),ptensors1(1)


class Ptensors1_ReLUFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,alpha):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        R.obj.add_ReLU(x.obj,alpha)
        ctx.x=x.obj
        ctx.alpha=alpha
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_ReLU_back(ctx.r.gradp(),ctx.alpha)
        return ptensors1.dummy(), None


class Ptensors1_inpFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.inp(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.y,g.item())
        ctx.y.add_to_grad(ctx.x,g.item())
        return ptensors1.dummy(), ptensors1.dummy()


class Ptensors0_diff2Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        ctx.x=x.obj
        ctx.y=y.obj
        return torch.tensor(x.obj.diff2(y.obj))

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_to_grad(ctx.x,g.item()*2.0)
        ctx.x.add_to_grad(ctx.y,-g.item()*2.0)
        ctx.y.add_to_grad(ctx.y,g.item()*2.0)
        ctx.y.add_to_grad(ctx.x,-g.item()*2.0)
        return ptensors1.dummy(), ptensors1.dummy()


class Ptensors1_concatFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        r=ptensors1(1)
        r.obj=_ptensors1.concat(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_concat_back(ctx.r,0)
        ctx.y.add_concat_back(ctx.r,ctx.x.get_nc())
        return ptensors1(1),ptensors1(1)


class Ptensors1_mprodFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_mprod(x.obj,y)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_mprod_back0(ctx.r.gradp(),ctx.y)
        return ptensors1(1), ctx.x.mprod_back1(ctx.r.gradp())


class Ptensors1_linearFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x,y,b):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),y.size(1),x.obj.get_dev())
        R.obj.add_linear(x.obj,y,b)
        ctx.x=x.obj
        ctx.y=y
        ctx.r=R.obj
        return R

    @staticmethod
    def backward(ctx,g):
        ctx.x.add_linear_back0(ctx.r.gradp(),ctx.y)
        return ptensors1.dummy(), ctx.x.linear_back1(ctx.r.gradp()), ctx.x.linear_back2(ctx.r.gradp())


class Ptensors1_Linmaps0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        R=ptens.ptensors0.zeros(x.obj.view_of_atoms(),x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_linmaps1to0(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to0_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)


class Ptensors1_Linmaps1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors1.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*2,x.obj.get_dev())
        ptens_base.add_linmaps1to1(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to1_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)


class Ptensors1_Linmaps2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        R=ptens.ptensors2.zeros(x.obj.view_of_atoms(),x.obj.get_nc()*5,x.obj.get_dev())
        ptens_base.add_linmaps1to2(R.obj,x.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_linmaps1to2_back(ctx.x.gradp(),ctx.r.gradp())
        return ptensors1(1)


class Ptensors1_Transfer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors0.zeros(atoms,x.obj.get_nc(),x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors1.dummy(), None, None


class Ptensors1_Transfer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors1.zeros(atoms,x.obj.get_nc()*2,x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors1.dummy(), None, None


class Ptensors1_Transfer2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,atoms,G):
        ctx.x=x
        R=ptens.ptensors2.zeros(atoms,x.obj.get_nc()*5,x.obj.get_dev())
        ptens_base.add_msg(R.obj,x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=R.obj
        ctx.G=G.obj
        return R
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_msg_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors1.dummy(), None, None


class Ptensors1_Unite1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors1(1)
        r.obj=ptens_base.unite1(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.unite1to1_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors1.dummy(), None


class Ptensors1_Unite2Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,G):
        r=ptens.ptensors2(1)
        r.obj=ptens_base.unite2(x.obj,G.obj)
        ctx.x=x.obj
        ctx.r=r.obj
        ctx.G=G.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.unite1to2_back(ctx.x.gradp(),ctx.r.gradp(),ctx.G)
        return ptensors1.dummy(), None


class Ptensors1_Outer0Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors1(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors1.dummy(), ptensors0.dummy()


class Ptensors1_Outer1Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,y):
        r=ptens.ptensors2(1)
        r.obj=ptens_base.outer(x.obj,y.obj)
        ctx.x=x.obj
        ctx.y=y.obj
        ctx.r=r.obj
        return r
        
    @staticmethod
    def backward(ctx,g):
        ptens_base.add_outer_back0(ctx.x.gradp(),ctx.r.gradp(),ctx.y)
        ptens_base.add_outer_back1(ctx.y.gradp(),ctx.r.gradp(),ctx.x)
        return ptensors1.dummy(), ptensors1.dummy()









