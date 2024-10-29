import ptens
import torch

A=ptens.ptensorlayer1.randn([[1,2,3],[3,5],[2]],3)
B=torch.relu(A)
S = ptens.subgraph.edge()
G1 = ptens.ggraph.random(5, 0.5)
G2 = ptens.ggraph.random(5, 0.5)
G3 = ptens.ggraph.random(5, 0.5)
G = ptens.batched_ggraph.from_graphs([G1, G2, G3])
C = ptens.batched_subgraphlayer1.from_ptensorlayers(G, S, [A,A,A])

D = ptens.batched_subgraphlayer1.from_ptensorlayers(G, S, [B,B,B])
print((C+D).__repr__())
