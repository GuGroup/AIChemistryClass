# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:16:45 2024

@author: user
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
dfs = pd.read_excel('data.xlsx', sheet_name='Sheet1')

data = dfs[['x','y']].to_numpy()

data[:,1] *= 0.0103642723 
m = LinearRegression().fit(data[:,0:1],data[:,1])

wc = m.coef_[0] 
bc = m.intercept_

ws = np.linspace(wc-2.5,wc+2.5,200)
bs = np.linspace(bc-10,bc+10,200)

ws,bs = np.meshgrid(ws,bs)

Js = ws[:,:,None]*data[:,0] - bs[:,:,None]

Js = (Js**2).sum(2)/(2*len(data))

fig, ax = plt.subplots()

CS = ax.contour(ws, bs, Js,levels=np.logspace(0,3,10))
#ax.clabel(CS, levels = CS.levels[1::8], inline=True, fontsize=8)
ax.clabel(CS,  inline=True, fontsize=8)
ax.set_xlabel('$w$',fontsize=20)
ax.set_ylabel('$b$',fontsize=20)
'''
Js [Js>400] = np.nan
#Js = np.clip(Js,0,600)
fig, ax = plt.subplots(figsize=(10, 10),subplot_kw={"projection": "3d"})
surf = ax.plot_surface(ws, bs, Js, cmap=cm['viridis'],linewidth=0, antialiased=True)

ax.set_xlabel('$w$',fontsize=20)
ax.set_ylabel('$b$',fontsize=20)
ax.set_zlabel('$J$',fontsize=20)
ax.set_zlim(0,600)
'''