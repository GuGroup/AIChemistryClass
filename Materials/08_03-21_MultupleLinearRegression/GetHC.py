# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:09:58 2024

@author: user
"""
import pandas as pd
from rdkit import Chem
from collections import Counter
import pubchempy
from tqdm import tqdm




df = pd.read_excel('alldata.xlsx', sheet_name='Sheet1')

data = df.values.tolist()

new_data = []
for d in data:
    if '[' not in d[0] and '=' not in d[0] and '#' not in d[0] and 'O' not in d[0] and \
        '1' not in d[0] and d[0] != 'C':
        new_data.append(d)


unique_groups = ['CH3','CH2','CH','C']
all_groups = []
new_new_data = []
for d in tqdm(new_data):
    mol = Chem.MolFromSmiles(d[0])
    datum = [pubchempy.get_compounds(d[0], namespace='smiles')[0].iupac_name,d[1]*0.0433641153087705]
    groups = []
    for atom in mol.GetAtoms():
        nH = atom.GetTotalNumHs()
        if nH == 0:
            groups.append('C')
        elif nH == 1:
            groups.append('CH')
        else:
            groups.append('CH%d'%nH)
    groups = Counter(groups)
    
    for g in unique_groups:
        datum.append(groups.get(g,0))
    new_new_data.append(datum)
    
df = pd.DataFrame(new_new_data)

df.columns = ['smiles','Hf[eV]','CH3','CH2','CH','C']



df.to_csv('Hydrocarbon.csv')


