# %% Import Libraries
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

folder='soc_rst_exp'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('opc ua client')
# %% Main body
pd.set_option("max_colwidth", 120)
read_ids={
        'AckRsrch':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bAckResearch",
        'RsrchMdApr':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bFreigabeResearch",
        'Bat' : "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bRmAnlageEin",
        # 'Grid':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bRmOn_Offgrid",
        'AlrmSt':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.iStatusFehler",
        'RsrchMdSt': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.iStatusResearch",
        'Pbat':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rWirkleistung",
        'SoC':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rSOC",
        'Psp':'ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.rSollWirkleistungAkt',
    
    #     'Status':"ns=2;s=Gateway.PLC1.65NT-06401-D001.PLC1.Dummy.strRead.strVentile.strY720.iStatusResearch",
    #     'DummyIstWert': "ns=2;s=Gateway.PLC1.65NT-06401-D001.PLC1.Dummy.strRead.strVentile.strY720.rIstwert",
        'ManMd':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bBetriebsartHand",
        'Alrm':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterienSystem.bError",
        'PT200': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strWechselrichter.strWechselrichterModbusBAT.rActivePower",
        # 'SoCmin' : "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterien.strBatterienSRC.rMinSOC",
        # 'SoCmax':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strRead.strBatterien.strBatterienSRC.rMaxSOC",
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

write_ids={
            'WD':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bWdResearch",
            'ReqRsrch': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bReqResearch",
            # 'OnOff':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bAnlageEin",
            'Manual':'ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bBetriebsartHand',
            # 'Grid': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bOn_Offgrid",
            # 'Quit':"ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.bQuittierung",
            'p_bat_sp': "ns=2;s=Gateway.PLC1.65NT-06402-D001.PLC1.microgrid.strWrite_L.strBatterienSystem.rSollWirkleistung"
            }
write_period=1
minute=60 # seconds
hour=60*minute
day=24*hour
Cbat=96 #kWh
p_bat=0
mmnt_list=[]
df_mmnt=pd.DataFrame(columns=list(read_ids.keys())+['timestamp'])
start=time.time()

last_mmnt=0.
sample_period=1
soc_lim=np.array([30,80])
soc_ref=(soc_lim[1]+soc_lim[0])/2
soc=get_last_soc()[0]
print('Last known SoC:',soc)
soc_max=0
soc_min=0
p_inv=0
p_load=0.
past=[]
t=0.
p_avg=0.
cl="clear"
p_bat_lim=1*Cbat #1C
p_std=p_bat_lim
soc_std=soc_lim[1]-soc_lim[0]
recovery_path=glob.glob(os.path.join(os.getcwd(),folder, 'recovery*'))
print(recovery_path,end='\n')
if len(recovery_path)>0:
    past=np.load(recovery_path[0])
    print('Recovered past is=',past)
    print('Mean=', past.mean(),'and  length',len(past)/60/60,'h')
elif np.any(np.isnan(past)) or past==[] :
    past=get_sum(dt=sample_period,d=1).values
    print('Read past is=',past)
    print('Mean=', past.mean(),'and  length',len(past)/60/60,'h')
#os.system(cl)

def p_soc_lim(g):
    dsoc=soc_ref-soc
    if dsoc>0:
        return (abs(dsoc)/soc_std*2)**g
    else:
        return -(abs(dsoc)/soc_std*2)**g


def RBC_soc_ref(x,p_bat_lim=Cbat):
    # RBC to set SoC to ref
    # x: float or int, exponential term 
    # p_bat_lim, float or int, power charge/discharge limit of the battery
    p_bound=p_bat_lim
    p_bat=p_bound*p_soc_lim(x)
    p_bat=np.clip(p_bat,-p_bat_lim,p_bat_lim)
    return p_bat
if __name__ == "__main__":
    # Run code only if Research Aknowledgment is 1
    rsrch_stat=get_rsrch_stat()[0]
    print('Research status code:',rsrch_stat)
    while rsrch_stat==2:
        print('time',datetime.datetime.now().strftime('%m %d %Y, %H:%M:%S'),' Research status blocked, code:',rsrch_stat)
        time.sleep(minute)
        rsrch_stat=get_rsrch_stat()[0]
    # Instance of OPCUA clinet subscrpition    
    opcua_client = opcua(user='AdrianPaeckel', password='SafeOpt_battery_2022')
    try:
        # Connect to OPCUA server
        opcua_client.connect()
        print('connection succesful')
        # Set read IDs for measurements
        df_Read= pd.DataFrame({'node': list(read_ids.values())})
        opcua_client.subscribe(json_Read=df_Read.to_json())
        # Get past day power data 
        p_load_list=[]
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
            except Exception as e:
                print('Connection ERROR while writing; reset operation activated!',e)
                print('\n')
                try: 
                # Reconnection if write process fails due to timeout or connection error
                    while get_rsrch_stat()[0]==2:
                        rsrch_stat=get_rsrch_stat()[0]
                        print('Research status blocked, code:',rsrch_stat)
                        time.sleep(5*minute)
                    opcua_client.disconnect()
                    time.sleep(1)
                    opcua_client.reset_sub(json_Read=df_Read.to_json())
                except Exception as e:
                    print('OPC UA disconnection ERROR',e)
            # Read measurement values     
            try:
                df_data = opcua_client.handler.df_Read
                df_data=df_data.set_index('node',drop=True).transpose()
                df_data=df_data[list(read_ids_.keys())]
                df_data=df_data.rename(columns=read_ids_)
                df_data=df_data.reset_index(drop=True)
                soc=df_data.loc[0,'SoC']
                # soc_max=df_data.loc[0,'SoCmax']
                # soc_min=df_data.loc[0,'SoCmin']
                p_inv=df_data.loc[0,'PT200']
                # Stat=df_data.loc[0,'Status']
                # AckRsrch=df_data.loc[0,'AckRsrch']
                # p_bat_sp=df_data.loc[0,'Psp']
                # dummy_var=df_data.loc[0,'DummyIstWert']
                # p_load=df_data.loc[0,['p_sol','p_sfw','p_vw','p_umr']].sum(axis=0)
                p_load=df_data.loc[0,'p_bypass']
                p_load_list.append(p_load)
            except Exception as e:
                print('Reading ERROR',e)
                print(p_bat)
                print('\n')
                continue 
            # Set active power setpoint
            try:         

                # print(len(past))
                # print(past[int(-15*minute//sample_period):].mean(),past.mean())
                p_avg=np.mean(past[int(-15*minute//sample_period):]) 
                if np.isnan(soc):
                    soc=get_last_soc()[0]
                    print('OPCUA SoC mmnt is NaN, last SoC from visualizer taken instead: ',soc)     
                p_bat_=RBC_soc_ref(1)
                if np.isnan(p_bat):
                    print(error_sentence2)
                else:
                    p_bat=p_bat_

            except Exception as e:
                print("RBC went wrong",e)
                print('Variables: ',p_bat)
                print('\n')

            try:
                #Do everything the needs to be done each 1s or sample period
                if ((time.time()-last_mmnt) > sample_period):
                    
                    last_mmnt=time.time()
                    t=last_mmnt-start
                    print(p_soc_lim(1))
                    # Filter p_load_list of non float values
                    p_load_list=list(filter(lambda x: isinstance(x,float),p_load_list))
                    # Get second average of p_load and reinitilize p_load_list
                    p_load_sec_avg=np.mean(p_load_list)
                    p_load_list=[]
                    # Pop last (24h before) power value and add new one
                    past=np.roll(past,-1)
                    if not np.isnan(p_load_sec_avg): past[-1]=p_load_sec_avg
                    else: 
                        past[-1]=past[-2]
                        print(error_sentence)

                    # Collect mmnt produced in RBC
                    mmnts=(p_bat,p_load_sec_avg,p_avg,round(t/hour,2))
                    if not all((isinstance(i,(float,int)) for i in mmnts)):
                        print('ERROR, not float or int:',list((not isinstance(i,(int,float)) for i in mmnts)))
                    # Collect mmnt from NEST server
                    df_data['timestamp']=datetime.datetime.now().strftime('%m %d %Y, %H:%M:%S')
                    for label,value in zip(['p_bat','p_l_s_avg','p_avg','exp_time'],mmnts):
                        df_data[label]=value
                    df_mmnt=pd.concat([df_data,df_mmnt],axis=0)
                    # Save each 30min collected data and reintialize dataframes/lists
                    if len(df_mmnt)*sample_period>2*minute:
                        save_data(df_mmnt,'df_mmnt',npy=False, folder=folder)
                        df_mmnt=pd.DataFrame(columns=list(read_ids.keys())+['timestamp'])
                    print(df_data)
                    try:
                        if not np.any(np.isnan(past)):
                            save_data(past,'recovery',date=False, folder=folder,verbose=0)
                        else:
                            print('ERROR past has NaN values',end='\n')
                    except Exception as e:
                        print('ERROR saving recovery data',e)
            except Exception as e:
                print("ERROR measurement",e)
                print('\n')

            time.sleep(write_period)
            
    except Exception as e:
        print("OPC UA disconnected",e)
        print(df_mmnt)
        print(mmnt_list)
        save_data(np.array(mmnt_list),'mmnt_list',npy=True,folder=folder)
        save_data(df_mmnt,'df_mmnt',npy=False,folder=folder)
        opcua_client.disconnect()
