#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

plt.style.use('science')

mpl.rcParams['figure.figsize'] = (12, 4)
mpl.rcParams['axes.grid'] = True
# # mpl.rcParams['axes.grid.axis']='y'
# mpl.rcParams['legend.markerscale']=8
# mpl.rcParams['text.usetex']=True

data=pd.read_csv('battery_dsoc_p.csv')
colors=['dimgray','red','blue']

plt.plot(data.power_t200_fil.values,data.shift(-1).dsoc.values,'.',color=colors[0],label=r'$\Delta s$ measurement',markersize=0.5,alpha=1)
# plt.plot(data.power_t200_fil,data.shift(-1).dsoc_m4,'.b',label='$\Delta s$ PWL ',markersize=1)
plt.plot(data.power_t200_fil.values,data.shift(-1).dsoc_m2.values,'.',color=colors[1],label='$\Delta s$ PWL',markersize=1)
plt.plot(data.power_t200_fil.values,data.shift(-1).dsoc_m5.values,'.',color=colors[2],label='$\Delta s$ ANN',markersize=1)
plt.ylabel(r'$\Delta$SoC[\%]')
plt.ylim([-2,2])
plt.xlabel(r'Active power [kW]')
plt.xlim([-150,150])
plt.legend(markerscale=8)
plt.yticks(np.arange(-2,2.1,1))
plt.savefig('dSoC_P.png')

