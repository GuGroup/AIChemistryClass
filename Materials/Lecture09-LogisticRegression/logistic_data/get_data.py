# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 21:22:54 2024

@author: user
"""

from matminer.datasets import get_available_datasets
from matminer.datasets import load_dataset
from pymatgen.core import Composition
import pandas as pd
get_available_datasets()

df = load_dataset("matbench_expt_is_metal")

data = df.to_numpy().tolist()

# find unique elements
elements = []
for d in data:
    c = Composition(d[0])
    if 'O' not in c: continue # only extract oxide
    elements += list(c.keys())

elements = [e for e in set(elements)]
elements_str = [str(e) for e in elements]

e_to_idx = {e:i+1 for i,e in enumerate(elements)}
csv_data = []
for d in data:
    c = Composition(d[0])
    if 'O' not in c: continue# only extract oxide
    row = [int(d[1])] + [0 for _ in elements]
    for k,v in c.fractional_composition.items():
        row[e_to_idx[k]] = v
    csv_data.append(row)

data = pd.DataFrame(csv_data)
data.columns = ['Metal'] + elements_str

data.to_csv('MetalbyComp.csv',index=False)