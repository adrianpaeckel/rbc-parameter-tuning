#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dense=tf.keras.models.load_model('/Users/adrianpaeckelripoll/rbc-parameter-tuning/ES_model/saved_models/p-dsoc_2-7-1.h5')

plt.style.use('science')

mpl.rcParams['figure.figsize'] = (12, 4)
mpl.rcParams['axes.grid'] = True
# # mpl.rcParams['axes.grid.axis']='y'
# mpl.rcParams['legend.markerscale']=8
# mpl.rcParams['text.usetex']=True


tf.keras.utils.plot_model(
    dense, to_file='ANN model.png', show_shapes=True, show_dtype=False,
    show_layer_names=False, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None
)

