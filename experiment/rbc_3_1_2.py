# This script is used to write the battery power setpoints and to read system measurements through the OPC UA server.
#Here we use the energy scheduling RBC 
#Written by Adrian Paeckel for the Master Thesis 'RBC parameter tuning: a Bayesian optimization approach'
#For any doubts, send email to adrainpaeckel@hotmail.com



# Import Libraries
from cmath import nan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from asyncore import read
from distutils.log import ERROR
import json
import logging
from tabnanny import verbose
from tkinter import E
from grpc import Status
from pyparsing import col
from opcuaclient_subscription import opcua
from opcuaclient_subscription import toggle
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
from mtfunc.helper import *
from mtfunc.datacqui import *
error_sentence='----------------ERROR P_load average is NaN-------------- \n'
error_sentence2='----------------ERROR P_bat_ is NaN-------------- \n'
#Set storage folder name
folder='ESRBC_exp0'
#Set RBC parameters
RBC_parameters=[0.4,-0.3]
#Set write/read period
write_period=0.7 #second
#Set logger level to INFO
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('opc ua client')
pd.set_option("max_colwidth", 120)
#Build a dict with system name and OPC UA IDs to read
read_ids={
        'AckRsrch':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bAckResearch",
        'RsrchMdApr':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bFreigabeResearch",
        'Bat' : "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bRmAnlageEin",
        'Grid':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bRmOn_Offgrid",
        'AlrmSt':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.iStatusFehler",
        'RsrchMdSt': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.iStatusResearch",
        'Pbat':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rWirkleistung",
        'SoC':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rSOC",
        'Psp':'ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rSollWirkleistungAkt',
        'ManMd':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bBetriebsartHand",
        'Alrm':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bError",
        'PT200': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strWechselrichter.strWechselrichterModbusBAT.rActivePower",
        'SoCmin' : "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterien.strBatterienSRC.rMinSOC",
        'SoCmax':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterien.strBatterienSRC.rMaxSOC",
        # 'p_sol':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU25E1_P001.rSumWirkleistung",
        # 'p_sfw':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU30E1_P001.rSumWirkleistung",
        # 'p_vw':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU23E1_P001.rSumWirkleistung",
        # 'p_umr':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU20E1_P001.rSumWirkleistung",
        'p_bypass':"ns=2;s=Gateway.PLC1.65NT-03007-D001.PLC1.ELM01.strELM01Read.strEnergiemessung.strP802.rSumWirkleistung",
        # 'p_grid':"ns=2;s=Gateway.PLC1.65NT-03007-D001.PLC1.ELM01.strELM01Read.strEnergiemessung.strP801.rSumWirkleistung",
        # 'p_m2c':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU12E1_P001.rSumWirkleistung",
        # 'p_dfab':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strEnergiemessung.strU33E1_P001.rSumWirkleistung",
        # 'p_move':"ns=2;s=Gateway.PLC1.65LK-06411-D001.PLC1.Move06411.strRead.strELM13.strP890.rSumWirkleistung"
        }
read_ids_= {v: k for k, v in read_ids.items()}
#Build dict with system name and OPC UA to write to
write_ids={
            'WD':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bWdResearch",
            'ReqRsrch': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bReqResearch",
            # 'OnOff':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bAnlageEin",
            'Manual':'ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bBetriebsartHand',
            # 'Grid': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bOn_Offgrid",
            # 'Quit':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bQuittierung",
            'p_bat_sp': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.rSollWirkleistung"}
#Set constants and initilize data containers
minute=60 # seconds
hour=60*minute
day=24*hour
#Battery capacity
Cbat=96 #kWh
p_bat=0.
mmnt_list=[]
df_mmnt=pd.DataFrame(columns=list(read_ids.keys())+['timestamp'])
start=time.time()
last_mmnt=0.
#Period of measurement on system
sample_period=10 #seconds
#Period for storing measurements in storage folder
save_period=5*minute
#Hard bounds on the battery SOC
soc_lim=np.array([35,75])
soc_ref=(soc_lim[1]+soc_lim[0])/2
#Get last SoC mmnt saved from the Visuallizer (in case OPC UA server down or NaN is given)
soc=soc=get_last_soc()[0]
soc_max=0
soc_min=0
p_inv=0
p_load=0.
past=[]
t=0.
p_avg=0.
#Set battery maximum power to 1C  
p_bat_lim=1*Cbat #1C
p_std=p_bat_lim
soc_std=soc_lim[1]-soc_lim[0]

P1=188 # buy price in euro/MWh
P2=170
P3=115
S1=53 #sell price in euro/MWh
def price_fun(k=None):
    price=DemandSignal(day,1,1)
    price.step_signal([0.29,0.33,0.54,0.71,0.87],[P3,P2,P1,P2,P1,P3])
    if k is None:
        return price.signal
    else:
        return price.signal[k]
# SoC reference tracking
def p_soc_lim(g):
#This term aims to reduce deviations from the reference SoC. 
#g: exponetial term, used to characterize the stiffness against deviations            
    dsoc=soc_ref-soc
    if dsoc>0:
        return (abs(dsoc)/soc_std*2)**g
    else:
        return -(abs(dsoc)/soc_std*2)**g

# def p_average_tracking():
#     p_avg_day=np.mean(past[int(-day//sample_period):])
#     return p_avg_day-p_avg

def lim_fun_2(price,expand,shift,p_bound=p_bat_lim/4):
    # bound=l*price+d
    #Time-dependent power limiting function
    norm_max_power=p_bound/(P1-P3)*price-p_bound/(P1-P3)*P3
    return expand*norm_max_power+shift*p_bound

def RBC_soc_ref(a,p_bat_lim=Cbat):
    #RBC to reset SoC to reference at the end of the day
    p_bound=p_bat_lim
    p_bat=p_bound*p_soc_lim(a)
    p_bat=np.clip(p_bat,-p_bat_lim,p_bat_lim)
    return p_bat    

def RBC_3_1_2(a,p_bat_lim=Cbat):
    # ES RBC
    # Always charge battery instead of injecting to the grid at 1C power (1kWh->1kW)
    #Par: expansion,shift, gamma (p_soc_lim)
    p_bound=p_bat_lim/4 #Normalizing constant
    day_time=datetime.datetime.now().hour*hour+datetime.datetime.now().minute*minute
    # Reset SoC at the beginning of the day
    if day_time<1*hour:
       # print('SoC restore')
        return RBC_soc_ref(0.7,p_bat_lim=Cbat)
    #Set maximal discharge power
    max_bound=lim_fun_2(price_fun(day_time),a[0],a[1],p_bound=p_bound)
    # Set high gammma value to avoid violating hard constraint while flexible capacity use
    gamma=11 # Do not change
    #Apply saturation
    p_bat=np.clip(-p_avg,-max_bound,p_bat_lim)
    if (p_avg>0)&(max_bound>0): p_bat= np.clip(p_bat,-p_avg,0)
    elif (p_avg>0): p_bat= p_bat
    elif (p_avg<=0)&(max_bound>0): p_bat= np.clip(p_bat,0,-p_avg)
    elif (p_avg<=0): p_bat= np.clip(p_bat,-max_bound,-p_avg)
    #Hard constraints on SoC
    if p_bat>0:
        p_bat=p_bat*(1+p_soc_lim(gamma))
    else:
        p_bat=p_bat*(1-p_soc_lim(gamma)) 
    return np.clip(p_bat,-p_bat_lim,p_bat_lim)

if __name__ == "__main__":
    # Run code only if Research Aknowledgment is not 2
    rsrch_stat=get_rsrch_stat()[0]
    print('Research status code:',rsrch_stat)
    while rsrch_stat==2:
        print('time',datetime.datetime.now().strftime('%m %d %Y, %H:%M:%S'),' Research status blocked, code:',rsrch_stat)
        time.sleep(minute)
        rsrch_stat=get_rsrch_stat()[0]
    #Read past load power measurements 
    past = get_sum(dt=sample_period, d=1).values
    print('Read past is=', past)
    print('Mean=', past.mean(), 'and  length', len(past) / 60 / 60, 'h')
    # User credentials
    opcua_client = opcua(user='AdrianPaeckel', password='SafeOpt_battery_2022')
    try:
        # Connect to OPCUA server
        opcua_client.connect()
        print('connection succesful')
        # Set read IDs for measurements
        df_Read= pd.DataFrame({'node': list(read_ids.values())})
        opcua_client.subscribe(json_Read=df_Read.to_json())
        p_load_list=[]
        attempt=0
        while True:
        # Write values to server: WD->toggle, AR->True, rStellWert active power ->p_bat
            try:
                df_Write = pd.DataFrame({'node': list(write_ids.values()),
                                        'value': [toggle(),
                                                     True,
                                                    # True,
                                                    True,
                                                    # False,
                                                    round(p_bat),
                                                ]})
                opcua_client.publish(json_Write=df_Write.to_json())
                attempt=0
            #If write process fails, check RsrchStatus code and attempt to connect again 5 times
            except Exception as e:
                print('Connection ERROR while writing; reset operation activated!',e)
                print('\n')
                try: 
                    while True:
                        rsrch_stat=get_rsrch_stat()[0]
                        if rsrch_stat!=2:
                            past = get_sum(dt=sample_period, d=1).values
                            print('Read past is=', past)
                            print('Mean=', past.mean(), 'and  length', len(past) / 60 / 60, 'h')
                            break

                        print('Research status blocked, code:',rsrch_stat)
                        time.sleep(5*minute)
                    attempt=attempt+1
                    print('Reconnection attempt:',attempt)
                    if attempt>5:
                        break
                    #opcua_client.disconnect()
                    time.sleep(5)
                    opcua_client.reset_sub(json_Read=df_Read.to_json())
                except Exception as e:
                    print('OPC UA disconnection ERROR',e)
            # Read measurement values     
            try:
                t1=time.time()
                df_data = opcua_client.handler.df_Read
                df_data=df_data.set_index('node',drop=True).transpose()
                df_data=df_data[list(read_ids_.keys())]
                df_data=df_data.rename(columns=read_ids_)
                df_data=df_data.reset_index(drop=True)
                soc=df_data.loc[0,'SoC']
                soc_max=df_data.loc[0,'SoCmax']
                soc_min=df_data.loc[0,'SoCmin']
                p_inv=df_data.loc[0,'PT200']
                # Stat=df_data.loc[0,'Status']
                # AckRsrch=df_data.loc[0,'AckRsrch']
                p_bat_sp=df_data.loc[0,'Psp']
                # p_load=df_data.loc[0,['p_sol','p_sfw','p_vw','p_umr']].sum(axis=0)
                p_load=df_data.loc[0,'p_bypass']
                p_load_list.append(p_load)
            except Exception as e:
                print('Reading ERROR',e)
                print('\n')
                continue 
            # Set active power setpoint
            try:  
                #Calculate load power 15min average       
                p_avg=np.mean(past[int(-15*minute//sample_period):])
                if np.isnan(soc):
                    soc=get_last_soc()[0]
                    print('OPCUA SoC mmnt is NaN, last SoC from visualizer taken instead: ',soc)           
                p_bat_=RBC_3_1_2(RBC_parameters)
                if np.isnan(p_bat):
                    print(error_sentence2)
                else:
                    p_bat=p_bat_

            except Exception as e:
                print("RBC went wrong",e)
                print('Variables: ',p_bat,past[-1])
                print('\n')

            try:
                #Do everything the needs to be done each sample period
                if ((time.time()-last_mmnt) > sample_period):
                    #Rst counter
                    last_mmnt=time.time()
                    t=last_mmnt-start
                    print('P_soc_lim: ',p_soc_lim(1))
                    # Filter p_load_list of non float values
                    p_load_list=list(filter(lambda x: isinstance(x,float),p_load_list))
                    # Get sample average of p_load and reinitilize p_load_list
                    p_load_sec_avg=np.mean(p_load_list)
                    p_load_list=[]
                    # Pop last (24h before) power value and add new one
                    past=np.roll(past,-1)
                    if not np.isnan(p_load_sec_avg): past[-1]=p_load_sec_avg
                    else: 
                        past[-1]=past[-2]
                        print(error_sentence)

                    # Collect measurements
                    day_time=datetime.datetime.now().hour*hour+datetime.datetime.now().minute*minute
                    mmnts=(p_bat,p_load_sec_avg,p_avg,price_fun(day_time), round(t/hour,2),RBC_parameters[0],RBC_parameters[1])
                    if not all((isinstance(i,(float,int)) for i in mmnts)):
                        print('ERROR, not float or int:',list((not isinstance(i,(int,float)) for i in mmnts)))
                    df_data['timestamp']=datetime.datetime.now().strftime('%m %d %Y, %H:%M:%S')
                    for label,value in zip(['p_bat','p_l_s_avg','p_avg','price','exp_time','par0','par1'],mmnts):
                        df_data[label]=value
                    df_mmnt=pd.concat([df_data,df_mmnt],axis=0)
                    # Save each 5min collected data and reintialize dataframes/lists
                    if len(df_mmnt)*sample_period>save_period:
                        save_data(df_mmnt,'df_mmnt',npy=False, folder=folder)
                        df_mmnt=pd.DataFrame(columns=list(read_ids.keys())+['timestamp'])
                    print(df_data,len(df_mmnt))
                    try:
                        if not np.any(np.isnan(past)):
                            save_data(past,'recovery',date=False, folder=folder,verbose=0)
                            save_data([day_time],'time_recovery',date=False, folder=folder,verbose=0)

                        else:
                            print('ERROR past has NaN values',end='\n')
                    except Exception as e:
                        print('ERROR saving recovery data',e)
            except Exception as e:
                print("ERROR measurement",e)
                print('\n')
            if time.time()-t1>0.4:
                print(time.time()-t1)
            time.sleep(write_period)
            
    except KeyboardInterrupt:
        # Exit script
        sys.exit("Script stopped by user")
    finally:
        print("OPC UA disconnected",e)
        print(df_mmnt)
        print(mmnt_list)
        save_data(np.array(mmnt_list),'mmnt_list',npy=True,folder=folder)
        save_data(df_mmnt,'df_mmnt',npy=False,folder=folder)
        opcua_client.disconnect()

