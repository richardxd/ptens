import torch
import ptens
import pytest
import ptens_base
from conftest import numerical_grad_sum, numerical_jacobian, get_atomspack_from_graph_factory_list, get_graph_list

from torch.autograd.gradcheck import gradcheck


def test_bug1(device):
    nnodes = 15
    graph = ptens.ggraph.random(nnodes, 0.5)
    subgraphs = [ptens.subgraph.trivial(), ptens.subgraph.edge()]
    node_values = torch.rand(nnodes, 1, requires_grad=True)

    node_attributes = ptens.subgraphlayer0.from_matrix(graph, ptens.subgraph.trivial(), node_values)

    for sg in subgraphs:
        gather_features = ptens.subgraphlayer0.gather(sg, node_attributes)
        result = torch.sum(gather_features)
        result.backward()

        # linmap_features = ptens.subgraphlayer0.linmaps(node_attributes)
        result = torch.sum(node_attributes)
        result.backward()

        check = gradcheck(ptens.subgraphlayer0.gather, (sg, node_attributes), eps=1e-3)
        assert check


def get_graph_atomspack_nc():
    ncs = [0, 1, 5]
    graphs = get_graph_list()
    atomspack_factory = get_atomspack_from_graph_factory_list()

    test_list = []
    
    for g in graphs:
        for atompack in [factory(g) for factory in atomspack_factory]:
            for nc in ncs:
                test_list.append((g, atompack, nc))
    return test_list

class TestGather(object):
    h = 1e-3

    def backprop_sum(self, cls, G, atoms, nc, device):
        x = cls.sequential(atoms, nc).to(device) + 1
        x = x.to(device)
        x.requires_grad_()
        print("x", x)
        atoms2 = G.subgraphs(ptens.subgraph.trivial())

        z = cls.gather(atoms2, x)

        zero = torch.zeros_like(z)
        l1_sum = torch.nn.L1Loss(reduction="sum")
        
        def loss_fn(x):
            z = cls.gather(atoms2, x)
            s = l1_sum(z, zero)
            # s = torch.sum(z)
            return s

        xgrad = torch.autograd.grad(outputs=loss_fn(x), inputs=x)[0]
        print("xgrad", xgrad)
        
        fn = lambda x: cls.gather(atoms2, x)
        xgrad2 = numerical_grad_sum(fn, x, self.h)
        print("xgrad2", xgrad2)
        assert torch.allclose(xgrad, xgrad2, rtol=1e-1, atol=1e-1)
        assert gradcheck(loss_fn, (x,), eps=self.h, rtol=1e-2, atol=1e-1, nondet_tol=1e-6)

        assert device in str(xgrad.device)

    def backprop_jac(self, cls, G, atoms, nc, device):
        x = cls.sequential(atoms, nc).to(device)
        x.requires_grad_()
        x = x.to(device)
        atoms2 = G.subgraphs(ptens.subgraph.trivial())

        def loss_fn(x):
            z = cls.gather(atoms2, x)
            return z

        xjac2 = numerical_jacobian(loss_fn, x, self.h)
        print("xjac2", xjac2)

        xjac = torch.autograd.functional.jacobian(loss_fn, x)
        print("xjac", xjac)
        xjac2 = xjac2.to(device)
        assert torch.allclose(xjac, xjac2, rtol=1e-2, atol=1e-2)
        assert gradcheck(loss_fn, (x,), eps=self.h, nondet_tol=1e-6)

        assert device in str(xjac.device)


    def _verify_input(self, G, atoms):
        N = max(G.adjacency_matrix().size())
        for i in range(len(atoms)):
            a = atoms[i]
            if len(a) > 0:
                assert max(a) < N        
        
    @pytest.mark.parametrize(("G", "atoms", "nc"), get_graph_atomspack_nc())
    def test_gather0_sum(self, G, atoms, nc, device):
        self._verify_input(G, atoms)
        self.backprop_sum(ptens.ptensorlayer0, G, atoms, nc, device)

    @pytest.mark.parametrize(("G", "atoms", "nc"), get_graph_atomspack_nc())
    def test_gather0_jac(self, G, atoms, nc, device):
        self._verify_input(G, atoms)
        self.backprop_jac(ptens.ptensorlayer0, G, atoms, nc, device)

    @pytest.mark.parametrize(('G', 'atoms', 'nc'), get_graph_atomspack_nc())
    def test_gather1_sum(self, G, atoms, nc, device):
        self._verify_input(G, atoms)
        self.backprop_sum(ptens.ptensorlayer1, G, atoms, nc, device)
        
    @pytest.mark.parametrize(('G', 'atoms', 'nc'), get_graph_atomspack_nc())
    def test_gather1_jac(self, G, atoms, nc, device):
        self._verify_input(G, atoms)
        self.backprop_jac(ptens.ptensorlayer1, G, atoms, nc, device)

    # @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
    # def test_gather2_sum(self, G, atoms, nc, device):
    #     self._verify_input(G, atoms)
    #     self.backprop_sum(ptens.ptensorlayer2, G, atoms, nc, device)
        
    # @pytest.mark.parametrize(('G', 'atoms', 'nc'), graph_atoms_list)
    # def test_gather2_jac(self, G, atoms, nc, device):
    #     self._verify_input(G, atoms)
    #     self.backprop_jac(ptens.ptensorlayer2, G, atoms, nc, device)
