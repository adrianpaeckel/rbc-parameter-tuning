import tensorflow as tf
from cmath import nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import pandas as pd
import numpy as np
import os 
import datetime 
from math import isclose
import math
import sys
import glob
dir_path=os.path.dirname(os.path.realpath(__file__))
# os.path.expanduser('~') 
os.chdir(dir_path)
sys.path.append('..')
import GPy
import safeopt
def current_cost_values():

def get_context():
    demand_model=tf.keras.models.load_model('../saved_models/demand_convtd_1dpowerdemand_Tfc_daysincos')
    PV_model=tf.keras.models.load_model('../saved_models/PVpower_200222')
def init_opt(folder):
    recovery_path=glob.glob(os.path.join(folder, 'recovery','optimizer'))
    print(recovery_path,end='\n')
    if len(recovery_path)>0:
        past=np.load(recovery_path[0])

    #For e_ch as context
    hp=[0.05**2,[0.19,0.12],4.4**2,[0.17,0.1],1,0.25,0.43]

    Vz=hp[0]
    Lz=hp[1]
    Vq=hp[2]
    Lq=hp[3]
    beta=hp[4]
    Lctxt_z=hp[5]
    Lctxt_q=hp[6]
    S_os_max=0

    noise_cst = 3e-5#cst_mmnt_noise.mean()/10
    noise_const =0.6 #const_mmnt_noise.mean()/10
    bounds = [(0., 4.),(-1,1)]
    parameter_set = safeopt.linearly_spaced_combinations(bounds, 480)
    # Define Kernel
    # for x_ in np.random.uniform(-0.5,-0.05,10):
    cost_kernel_ = GPy.kern.RBF(input_dim=len(bounds), variance=Vz, lengthscale=[Lz[0],Lz[1]], ARD=True,active_dims=[0,1])
    cost_context_kernel=GPy.kern.RBF(input_dim=1, variance=1, lengthscale=Lctxt_z, ARD=True,active_dims=[2],name='context')
    cost_kernel=cost_kernel_*cost_context_kernel
    const_kernel_=GPy.kern.RBF(input_dim=len(bounds), variance=Vq, lengthscale=[Lq[0],Lq[1]], ARD=True,active_dims=[0,1])
    const_context_kernel=GPy.kern.RBF(input_dim=1, variance=1, lengthscale=Lctxt_q, ARD=True,active_dims=[2],name='context')
    const_kernel=const_kernel_*const_context_kernel


    y0= np.atleast_2d(cost.send(None))

    cost_gp = GPy.models.GPRegression(x,y0[:,0,None], cost_kernel, noise_var=noise_cst)
    const_gp= GPy.models.GPRegression(x,y0[:,1,None], const_kernel, noise_var=noise_const)
    gp=[cost_gp,const_gp]
    # The optimization routine
    opt = safeopt.SafeOpt(gp,parameter_set=parameter_set,fmin=[-np.inf,-S_os_max], threshold=np.array([-np.inf,-50]),beta=beta,num_contexts=1)
    return opt