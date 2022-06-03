# This script uses the multiprocessing tool the perform a grid evaluation of the parameter domain. This code is just an uncommented snap of the /simulation/ESRBC_sim.ipynb. Please refer to the latter if some code sections are unclear.
import os
from time import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import sys
dir_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.insert(0,'..')
from mtfunc.helper import *
import matplotlib as mpl
import matplotlib.pyplot as plt


hour=60
day=24*hour
C=96 #kWh
soc_lim=np.array([30,80])
p_bat_lim=1*C #1C 
p_std=p_bat_lim
soc_std=soc_lim[1]-soc_lim[0]# do not change unless necesary
soc_ref=(soc_lim[1]+soc_lim[0])/2
dt=1
S_os_max=0 # max overshoot of SoC; ex. SoC_max=soc_lim[1]+S_os_max
print(p_std,soc_std)

import tensorflow as tf
from tensorflow import keras

phib=np.asarray(pd.read_csv('../ES_model/PWLmodel_par_opti.csv',index_col='Unnamed: 0'))

data=get_data('model_data/model_spring_data_2022_03_16_0745.csv',npy=False)
data=data.set_index(pd.to_datetime(data.index))

def batt_model_generator(x,soc_i,mtype='PWL',model=None):
# Creates generator of the battery model
# IC for soc
# first siganl out is IC when initialized (send(None))
# send(power) for next signal
    if mtype=='PWL':
        cut1=x[5]
        cut2=x[6]
        def soc_func(soc,p):
            dsoc=x[0]+x[1]*max(cut1,p)+x[2]*max(0,p)+x[3]*max(cut2,p)+x[4]*p
            if dsoc>x[9]: dsoc=x[7]*dsoc
            if dsoc<x[10]: dsoc=x[8]*dsoc
            return soc+dsoc
    if mtype=='ANN':
        assert model is not None                            
        def soc_func(soc,p):
            dsoc=model.predict(np.atleast_2d([p])).flatten()
            return soc+dsoc
        
    soc=soc_i
    p=0
    while True:
        p=yield soc
        soc=soc_func(soc,p)[0]

P1=188 # buy price in euro/MWh
P2=170
P3=115
S1=53 #sell price in euro/MWh
B1=1
def price_fun(k=None):
    price=DemandSignal(day,dt,1)
    price.step_signal([0.29,0.33,0.54,0.71,0.87],[P3,P2,P1,P2,P1,P3])
    if k is None:
        return price.signal
    else:
        return price.signal[k]

# Uncomment to use other utility function
# B1=1
# P1=200
# P3=150
# S1=50
# def price_fun(k=None):
#     price=DemandSignal(day,dt,1)
#     price.step_signal([0.2,0.4,0.6,0.7,0.9],[P1,P3,P1,P3,P1,P3])
#     if k is None:
#         return price.signal
#     else:
#         return price.signal[k]

# %%
def batt_control_generator(dt,a,soc_lim,p_bat_lim,N,past,p_set=[0,4],T=day):
# Creates a generator of the BMS
# P_bat >0 for charging 
# a: array, RBC parameters
# soc_lim: array, SoC limits (0,100)
# p_bat_lim: battery charge and discharge limits in kW
# N: array, moving avarage window widths for peak shaving in minutes
# dt= sampling period in minutes

        k=0
        soc_ref=50
        soc=50
        p_avg=past[-N[0]//dt:].sum()/past[-N[0]//dt:].size
        def p_cap():
            p_2=past[-15//dt:].sum()/past[-15//dt:].size
            p_1=past[-1//dt:].sum()/past[-1//dt:].size
            return p_2-p_1
        def p_soc_lim(g):
            dsoc=soc_ref-soc
            if dsoc>0:
                return (abs(dsoc)/soc_std*2)**g
            else:
                return -(abs(dsoc)/soc_std*2)**g
        def p_24h_average_tracking():
            p_avg_day=past[-N[1]//dt:].sum()/past[-N[1]//dt:].size
            return p_avg_day-p_avg
    
        def lim_fun_2(price,expand,shift,p_bound=C/6):
            # bound=l*price+d
            norm_max_power=p_bound/(P1-P3)*price-p_bound/(P1-P3)*P3
            return expand*norm_max_power+shift*p_bound
        # PS RBC
        def control_3_3_0():
            #Peak-shaving with flat bound
            #Same charge and discharge bound
            #Par: expansion, gamma, (allow take power from the grid )
            p_bound=p_bat_lim
            max_dis=a[0]*p_bound
            p_bat=np.clip(p_24h_average_tracking(),-max_dis,max_dis)
            p_bat=p_bat+max_dis*p_soc_lim(a[1])
            if p_avg>0 :
                return np.clip(p_bat,-p_avg,None)
            else:
                return np.clip(p_bat,0,-p_avg)                
        # ES RBC
        def control_3_1_2():
            
            p_bound=C/4
            max_bound=lim_fun_2(price_fun(k),a[0],a[1],p_bound=p_bound)
            gamma=11
            p_bat=np.clip(-p_avg,-max_bound,p_bat_lim)

            if (p_avg>0)&(max_bound>0): p_bat= np.clip(p_bat,-p_avg,0)
            elif (p_avg>0): p_bat= p_bat
            elif (p_avg<=0)&(max_bound>0): p_bat= np.clip(p_bat,0,-p_avg)
            elif (p_avg<=0): p_bat= np.clip(p_bat,-max_bound,-p_avg)
            if p_bat>0:
                p_bat=p_bat*(1+p_soc_lim(gamma))
            else:
                p_bat=p_bat*(1-p_soc_lim(gamma)) 
            return p_bat

                
        while True:
            p_avg=past[-N[0]//dt:].sum()/past[-N[0]//dt:].size

            if k*dt/day<24/24:
                # p_bat=control_3_1_2()
                #Uncomment to use the PS RBC
                p_bat=control_3_3_0()                
            else:
                p_bat=p_bat_lim*p_soc_lim(1)
            if soc>80:
                p_bat=min(0,p_bat)
            if soc<20:
                p_bat=max(0,p_bat)
#             Max charge/discharge power saturation 1C
            p_bat=np.clip(p_bat,-p_bat_lim,p_bat_lim)
                
            soc,soc_ref,p_dem,k=yield p_bat,p_cap()# gets soc, soc_ref, p_dem ;; p_bat out
            past=np.append(past,p_dem)


def add_sin_noise(x,period,amp,rnd=0.5,k=None):
    if k is None:
        t=np.arange(len(x))*dt
        phase_noise=2*rnd*np.pi*np.random.rand(len(x))
        w=2*np.pi/period
        n=amp*np.sin(w*t+phase_noise)
        return x+n
    else:
        t=k*dt
        phase_noise=2*rnd*np.pi*np.random.rand()
        w=2*np.pi/period
        n=amp*np.sin(w*t+phase_noise)
        return x+n  

def plant_cost(x,T=1,p_load=data.power_load.values,N_lim=[15,day],soc_lim=soc_lim,soc_softc=[35,80],soc_i=50,phib=phib,plot=False,plot_cost=False,plot_constrain=False,
               soc_ref=soc_ref,dt=1,p_set=[0,4],noise=False):
# Takes new control parameters, simulates a day and outputs cost and contrain functions
    
# x: array, RBC parameters
# T: optimization period in days
# soc_lim: array, SoC limits 
# soc_ref: reference for SoC controler
# dt: sampling period in minutes
# N_lim: array, moving avarage window widths for peak shaving 

    dt=dt
    T=T*day//dt
    p_load=p_load[:T+N_lim[1]]
    bc=batt_control_generator(dt,x,soc_lim,p_bat_lim,N_lim,p_load[:N_lim[1]//dt],p_set=p_set,T=T)
    p_load=p_load[N_lim[1]//dt:]
    bm=batt_model_generator(phib,soc_i)
    soc=[bm.send(None)] #Initialize battery model ->SoC_0
    bc.send(None)
    p_bat=[] #Initiliazation controller
    p_grid=[]
    if noise:
        for k,po in enumerate(p_load):
            p_bat+=[bc.send((add_sin_noise(soc[-1],hour,0.5,0.5,k),soc_ref,add_sin_noise(po,hour,load_noise,0.5,k),k))[0]]
            p_cap=bc.send((add_sin_noise(soc[-1],hour,0.5,0.5,k),soc_ref,add_sin_noise(po,hour,load_noise,0.5,k),k))[1]
            soc+=[bm.send(add_sin_noise(p_bat[-1],hour,bat_noise,1,k))]
            p_grid+=[p_bat[-1]+po+p_cap]
    else:
        for k,po in enumerate(p_load):
            p_bat+=[bc.send((soc[-1],soc_ref,po,k))[0]]
            p_cap=bc.send((soc[-1],soc_ref,po,k))[1]
            soc+=[bm.send(p_bat[-1])]
            p_grid+=[p_bat[-1]+po+p_cap]                   
    soc_f=soc[int(T*23.5/24)]
    soc=np.array(soc[:-1])
    p_grid=np.array(p_grid)
    p_bat=np.array(p_bat)
    def cost_f(plot=plot_cost):
        #SoC reference metric
        soc_norm=np.abs(np.power((soc-soc_ref)/soc_std,3))
        soc_norm=soc_norm
        #Monetary metric
        def r_price_sell_fun():    
            def price_sell_map_fun(p_bat,p_load,f_buy,f_sell):
                if (-p_bat<=p_load)&(p_load>=0):
                    return -p_bat*f_buy 
                if (-p_bat>p_load)&(p_load>=0):
                    return -f_sell*(p_bat+p_load)+f_buy*p_load 
                if (-p_bat<=p_load)&(p_load<0):
                    return -f_buy*(p_bat+p_load)+f_sell*p_load 
                if (-p_bat>p_load)&(p_load<0):
                    return -p_bat*f_sell 
                else: return 0
            return np.array(list(map(price_sell_map_fun,p_bat,p_load,price_fun(),S1*np.ones(T))))/hour*dt*1e-3 # [euro]
        r_price_sell=r_price_sell_fun()
        # Discharge power metric
        g_b_dis=np.maximum(np.zeros_like(p_bat),np.sign(-p_bat))*p_bat*B1/hour*dt*1e-3 # cost only when discharging in euro
        # Battery use metric
        g_b=-np.abs(p_bat)*B1/hour*dt*1e-3 # battery cost ch/dis in euro
       
        return  -soc_norm.sum()/T,r_price_sell.sum(),g_b_dis.sum(),g_b.sum()

    def local_soc_min(s,si):
        if min(s)<si:
            return min(s)
        else:
            order=hour/dt
            base_idx=signal.argrelmin(s,order=int(order))[0]
            if len(s[base_idx])>0:
                return min(s[base_idx])
            else :
                return s[-1]


    return cost_f(),soc_f,local_soc_min(soc,soc_i) 
max_eval=1000
days=len(data)//day-1
rnd_opt_cst=[]
rnd_opt_par=[]
soc_lim=[30,80]
soc_softc=[32,80]
soc_ref=(soc_lim[1]+soc_lim[0])/2
S_os_max=0
S_f_sp=55
w=[1,50]
print(max_eval,soc_ref,soc_lim,S_f_sp,S_os_max)
import nlopt as nl
def func(d):
    max_eval=1000
    def cost(x,gradient=0):
        c=plant_cost(x,p_load=data.power_load.values[d*day:],T=1,N_lim=[15,day],soc_lim=soc_lim,soc_softc=soc_softc,soc_ref=soc_ref,soc_i=50,phib=phib,noise=False)[0]
        cost=w[0]*c[3]+w[1]*c[0]
        return -cost
    def constraint(x,gradient=0):
        q=plant_cost(x,p_load=data.power_load.values[d*day:],T=1,N_lim=[15,day],soc_lim=soc_lim,soc_softc=soc_softc,soc_ref=soc_ref,soc_i=50,phib=phib,noise=False)[1:]
#         print( -(q[-1]-S_f_sp)-S_os_max)
        return -(q[0]-S_f_sp)
#     Global optimization
    opt = nl.opt(nl.GN_ISRES, 2)
    opt.set_lower_bounds([0.,1.])
    opt.set_upper_bounds([2.,11.])
    opt.set_min_objective(cost)    
    opt.add_inequality_constraint(lambda x,grad: constraint(x,grad), 1e-8)
    opt.set_xtol_rel(1e-1)
    opt.set_maxeval(max_eval)
    a_ = opt.optimize([ 1., 1.])
    res_code=opt.last_optimize_result()
    print(d,res_code)

#     Local optimization

    code=5
    tol=1e-3
    while code==5:
        if res_code!=4:
            a=[1.,1.]
        else:
            a=a_
        a=[1.,1.]
        opt = nl.opt(nl.LN_COBYLA, 2)
        opt.set_lower_bounds([0.,1.])
        opt.set_upper_bounds([2.,11.])
        opt.set_min_objective(cost)    
        opt.add_inequality_constraint(lambda x,grad: constraint(x,grad), 1e-8)
        opt.set_xtol_rel(tol)
        opt.set_maxeval(max_eval)
        max_eval=max_eval-100
        tol=tol*1e1
        if tol>1e-1:
            break
        a = opt.optimize(a)
        minf = opt.last_optimum_value()
        code=opt.last_optimize_result()

    print('day: ',d)
    print("optimum at ", a[0], a[1])
    print("minimum value = ", minf)
    print("result code = ",code )
    return a,minf,code
    

if __name__ == '__main__':
    import time
    from multiprocessing import Pool
    
    start_time = time.time()

    
    days=30
    try:
        with Pool(os.cpu_count()-1) as p:
            R=p.map(func, range(days))
    except Exception as e:
        print(e)
        print("Process halted exception --- %s seconds ---" % (time.time() - start_time))
        Z=np.vstack([z[0] for z in R])
        Q=np.vstack([z[1] for z in R])
        M=np.vstack([z[2] for z in R])
        print(Z.shape)
        print(Q.shape)
        save_data(Z,'mrnd_opt_par_winter_3_3_0',folder='NL_opt')
        save_data(Q,'mrnd_opt_cst_winter_3_3_0',folder='NL_opt')
        save_data(M,'mrnd_opt_code_winter_3_3_0',folder='NL_opt')
        meta_data=[B1,soc_lim,soc_softc,'gbdis_gb10e-2_qs',Cdev]
        print(meta_data)
        save_data(meta_data,'mrnd_opt_metadata_winter_3_3_0',folder='NL_opt')

    finally:

        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        Z=np.vstack([z[0] for z in R])
        Q=np.vstack([z[1] for z in R])
        M=np.vstack([z[2] for z in R])
        print(Z.shape)
        print(Q.shape)
        save_data(Z,'mrnd_opt_par_winter_3_3_0',folder='NL_opt')
        save_data(Q,'mrnd_opt_cst_winter_3_3_0',folder='NL_opt')
        save_data(M,'mrnd_opt_code_winter_3_3_0',folder='NL_opt')
        meta_data=[B1,soc_lim,soc_softc,'gbdis_gb10e-2_qs',Cdev]

        print(meta_data)
        save_data(meta_data,'mrnd_opt_metadata_winter_3_3_0',folder='NL_opt')