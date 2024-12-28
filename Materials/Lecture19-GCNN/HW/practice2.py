import json
from rdkit import Chem
import numpy as np
import torch
data = json.load(open('SmilesAndG.json'))

def ProcessDatum(d):
    mol = Chem.MolFromSmiles(d[0]) 
    mol = Chem.AddHs(mol)
    atoms = ['H','C','N','O','F']
    v = []
    for atom in mol.GetAtoms():
        oh = [0 for _ in atoms]
        oh[atoms.index(atom.GetSymbol())] = 1
        v.append(oh)
    v = torch.tensor(v,dtype=torch.float)
    c = []
    for bond in mol.GetBonds():
        c.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        c.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]) # torch_geometric requires bi_directional graph
    c = torch.tensor(c,dtype=torch.long).T # torch wants transposed matrix
    y = torch.tensor([d[1]*2625.5]) # convert to kJ/mol
    return v,c,y

import torch
from torch_geometric.data import Data, Dataset
import os
from tqdm import tqdm

class SmilesData(Dataset):
    def __init__(self, root='./'):
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        data = json.load(open(os.path.join(root,'SmilesAndG.json')))
        self.data = []
        for d in tqdm(data):
            v,c,y = ProcessDatum(d)
            self.data.append(Data(x=v,edge_index=c,y=y))
        # The training is significantly better if atomic contribution is calculated and subtracted.
        x = []
        y = []
        for d in self.data:
            x.append(d.x.sum(0))
            y.append(d.y)
        x = torch.stack(x)
        y = torch.cat(y)
        self.atomistic_contribution = torch.linalg.lstsq(x,y).solution
        y = y - x@self.atomistic_contribution
        for d,yy in zip(self.data,y):
            d.y = yy

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
data = SmilesData()