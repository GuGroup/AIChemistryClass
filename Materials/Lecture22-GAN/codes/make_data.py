# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:46:13 2022

@author: user
"""
from pymatgen.ext.matproj import MPRester
import numpy as np
from ase.data import chemical_symbols
from pymatgen.core import Composition
import csv

with MPRester("gv5swR0lIWsfLsGk") as m:
    docs = m.query({},{'pretty_formula':True,'icsd_ids':True})
    raw_data = [(doc['pretty_formula'],doc['icsd_ids']) for doc in docs]

new_data = {}
for f,v in raw_data:
    if len(v) != 0:
        c = Composition(f)
        new_data[str(c)] = dict(c.fractional_composition)

sym_map = {}
for i,s in enumerate(chemical_symbols):
    sym_map[s] = i -1

data = np.zeros((len(new_data),len(chemical_symbols)-1))
names = []
for i,(name,d) in enumerate(new_data.items()):
    names.append(name)
    for k,v in d.items():
        data[i,sym_map[str(k)]] = v

data_to_copy = [['formula']+chemical_symbols[1:]]
for n, r in zip(names,data):
    data_to_copy.append([n]+r.tolist())

with open('data.csv','w',newline='') as f:
    writer = csv.writer(f)
    for l in data_to_copy:
        writer.writerow(l)