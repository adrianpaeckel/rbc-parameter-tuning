#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 4)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.grid.axis']='y'

data=pd.read_csv('/Users/adrianpaeckelripoll/rbc-parameter-tuning/figure_scripts/boxplot_battery_metrics/metrics.csv',index_col='index')

model_label=data.columns.values[:5]
model_label=['soc_m1','soc_m2','soc_m4','soc_m5']
model_names=['M1','M2','M3','ANN']


data.columns=[col[:6] for col in data.columns.values]
metric='r2'
plt.figure(1)
for i,col in enumerate(model_label): 
    plt.subplot(121)
    plt.boxplot(data[col].loc[metric].values,showfliers=False,labels=[model_names[i]],positions=[i])
    plt.ylabel(str.upper(metric))
plt.locator_params(axis="y", nbins=5)       
metric='mae'
for i,col in enumerate(model_label): 
    plt.subplot(122)
    plt.boxplot(data[col].loc[metric].values,showfliers=False,labels=[model_names[i]],positions=[i])
    plt.ylabel(str.upper(metric))
plt.locator_params(axis="y", nbins=5)   
plt.savefig('R2_mae.png')
    

