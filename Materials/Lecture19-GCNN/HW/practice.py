# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:05:16 2024

@author: user
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, aggr
from torch_geometric.nn.pool import global_mean_pool

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [-12]], dtype=torch.float)

edge_attr = torch.tensor([[0],
                           [1]], dtype=torch.float)
data1 = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=1)


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [2]], dtype=torch.float)
edge_attr = torch.tensor([[0],
                           [1]], dtype=torch.float)
data2 = Data(x=x, edge_index=edge_index,y=2,edge_attr=edge_attr)

Loader = DataLoader([data1,data2],batch_size=2)


batch = next(iter(Loader))

wow = GCNConv(1,2)

x = wow(batch.x,batch.edge_index)

mean_aggr = aggr.MeanAggregation()

mean_aggr(x,batch.batch)

global_mean_pool(x,batch.batch)