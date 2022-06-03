


from calendar import week
import sys
import os
from tracemalloc import start
from turtle import back
import restclient
import pandas as pd
import datetime 
def get_last_soc():
    now=datetime.datetime.now()
    bck=datetime.timedelta(minutes=5)
    start_date=now-bck
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=now.strftime('%Y-%m-%d %H:%M:%S')

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={
        'soc_max':'40200104',
    }

    df_list=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    return data_read.iloc[-1]
def get_rsrch_stat():
    now=datetime.datetime.now()
    bck=datetime.timedelta(minutes=5)
    start_date=now-bck
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=now.strftime('%Y-%m-%d %H:%M:%S')

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={
        'Research Status':'40200016'
    }

    df_list=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    return data_read.iloc[-1]

def get_resampled(dt=1,d=7):    
# gets historical data from nest server
# dt: resampling period in seconds
# d:  days past, default is a week,eg 7

    now=datetime.datetime.now()
    bck=datetime.timedelta(days=d)
    start_date=now-bck
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=now.strftime('%Y-%m-%d %H:%M:%S')

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={
    # # ----- EL. ENERGY -----
    # # SFW unit
    # 'power_sfw':'42110000',
    # # DFAB unit
    # 'power_dfab':'42190139',
    # # MOVE unit
    # 'power_move':'401190130',
    # # SOLACE unit
    # 'power_sol':'42160255',
    # #Vision and Wood
    # 'power_vw':'42120000',
    # #Urban Mining and Recycling
    # 'poewr_umr':'42150423',
    # #Meet2Create
    # 'power_m2c':'42140000',
    # # ELM01 Ehub power mmnt
    # 'power_ehub':'401190042',
    'power_bypass_ehub':'401190064',
    # 'power_bypass_estore':'401190086',
    # # ----SOLAR POWER------
    # # SFW unit
    # 'power_pv_sfw_t100':'42110038',
    # 'power_pv_sfw_t101':'42110078',
    # 'power_pv_sfw_t102':'42110118',
    # # DFAB unit
    # 'power_pv_meter_dfab':'42190053',
    # 'power_pv_dfab_t100':'421110102' ,
    # # MOVE building
    # # 'power_pv_move_t107':'3210295',
    # # 'power_pv_move_t108':'3210335',
    # 'power_pv_move':'4180039',
    # 'power_pv_move_t100':'40200397',
    # 'power_pv_move_t100_d':'40200422',
    # # SOLACE unit
    # 'power_pv_sol_t100':'42160278',
    # # Battery
    # 'current':'40200103',
    # 'voltage':'40200102',
    # 'soc_max':'40200104',
    # 'soc_min':'40200105',
    # # Battery power mmnts
    # 'power_t200':'40200263', # AC inverter mmnt
    # 'power_bat_sp':'40200005', # G200 active power setpoint
    # 'power_bat_g200':'40200017', #G200 active power mmnt
    # # power_inverter_DC:'40200265' #DC inverter mmnt
    # 'power_bat_meter':'401190152', # ELM01 active power mmnt
    # # # 'research_mode' : '40200011', # 1:research,0: controller
    # # Weather 
    # 'global_radiation':'3200008',
    # 'global_radiation_sol':'42160159',
    # 'global_radiation_fc':'402190002',
    # 'outside_temperature':'3200000',
    # 'outside_temperature_fc':'402190000',
    # 'wind_speed':'3200004',
    # 'rh_outside':'3200002',
    # 'rh_outside_fc':'402190001',
    # 'solar_elevation':'3200020',
    # 'solar_azimuth':'3200021'
    }

    df_list=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            data=data.resample(f'{dt}S').mean().interpolate().dropna()
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    return data_read
def get_sum(dt=1,d=7):
    # Get sum of exp. units (EMPA) in metada list in get_resampled method
    df=get_resampled(dt,d)
    return df.sum(axis=1)
def get_PV(dt=60,d=1):    

# gets historical data from nest server
# dt: resampling period in seconds
# d:  days past, default is a week,eg 7

    now=datetime.datetime.now()
    bck=datetime.timedelta(days=d)
    start_date=now-bck
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=now.strftime('%Y-%m-%d %H:%M:%S')

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={
    # # ----- EL. ENERGY -----
    # # SFW unit
    # 'power_sfw':'42110000',
    # # DFAB unit
    # 'power_dfab':'42190139',
    # # MOVE unit
    # 'power_move':'401190130',
    # # SOLACE unit
    # 'power_sol':'42160255',
    # #Vision and Wood
    # 'power_vw':'42120000',
    # #Urban Mining and Recycling
    # 'poewr_umr':'42150423',
    # #Meet2Create
    # 'power_m2c':'42140000',
    # # ELM01 Ehub power mmnt
    # 'power_ehub':'401190042',
    # 'power_bypass_ehub':'401190064',
    # 'power_bypass_estore':'401190086',
    # # ----SOLAR POWER------
    # SFW unit
    'power_pv_sfw_t100':'42110038',
    'power_pv_sfw_t101':'42110078',
    'power_pv_sfw_t102':'42110118',
    # DFAB unit
    'power_pv_meter_dfab':'42190053',
    # 'power_pv_dfab_t100':'421110102' ,
    # MOVE building
    # 'power_pv_move_t107':'3210295',
    # 'power_pv_move_t108':'3210335',
    'power_pv_move':'4180039',
    'power_pv_move_t100':'40200397',
    # 'power_pv_move_t100_d':'40200422',
    # SOLACE unit
    'power_pv_sol_t100':'42160278',
    # # Battery
    # 'current':'40200103',
    # 'voltage':'40200102',
    # 'soc_max':'40200104',
    # 'soc_min':'40200105',
    # # Battery power mmnts
    # 'power_t200':'40200263', # AC inverter mmnt
    # 'power_bat_sp':'40200005', # G200 active power setpoint
    # 'power_bat_g200':'40200017', #G200 active power mmnt
    # # power_inverter_DC:'40200265' #DC inverter mmnt
    # 'power_bat_meter':'401190152', # ELM01 active power mmnt
    # # # 'research_mode' : '40200011', # 1:research,0: controller
    # # Weather 
    # 'global_radiation':'3200008',
    # 'global_radiation_sol':'42160159',
    # 'global_radiation_fc':'402190002',
    # 'outside_temperature':'3200000',
    # 'outside_temperature_fc':'402190000',
    # 'wind_speed':'3200004',
    # 'rh_outside':'3200002',
    # 'rh_outside_fc':'402190001',
    # 'solar_elevation':'3200020',
    # 'solar_azimuth':'3200021'
    }

    df_list=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            data=data.resample(f'{dt}S').mean().interpolate().dropna()
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    data_read=data_read.abs()
    return data_read.sum(axis=1)
def get_weather_fc(dt=60,d=1):


# gets historical data from nest server
# dt: resampling period in seconds
# d:  days past, default is a week,eg 7

    now=datetime.datetime.now()
    bck=datetime.timedelta(days=1)
    start_date=now-bck*(d-1)
    end_date=now+bck

    start_date_solar=now-bck*d
    end_date_solar=now
    
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_solar=end_date_solar.strftime('%Y-%m-%d %H:%M:%S')
    start_date_solar=start_date_solar.strftime('%Y-%m-%d %H:%M:%S')
    print(start_date,end_date)

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={
    # # Weather 
    # 'global_radiation':'3200008',
    # 'global_radiation_sol':'42160159',
    
    # 'outside_temperature':'3200000',
    'outside_temperature_fc':'402190000',
    # 'wind_speed':'3200004',
    # 'rh_outside':'3200002',
    'rh_outside_fc':'402190001',
    # 'solar_elevation':'3200020',
    # 'solar_azimuth':'3200021'
    'global_radiation_fc':'402190002',
    }

    meta_solar={
        'solar_elevation':'3200020',
        'solar_azimuth':'3200021'
    }
    df_list=[]
    df_list2=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            data=data.resample(f'{dt}S').mean().interpolate().dropna()
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    for key in list(meta_solar.keys())[:]:
            try:
                data = rest.read(df_data=pd.DataFrame(columns=[meta_solar[key]]),
                                                        startDate=start_date_solar,
                                                        endDate=end_date_solar)
                data.columns=[key]
                data=data.resample(f'{dt}S').mean().interpolate().dropna()
                df_list2+=[data.copy()]
            except Exception as e: 
                print(key,'did not work',e)            
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    data_read_solar=pd.concat(df_list2,axis=1)
    data_read_solar=data_read_solar.dropna()


    return pd.concat([data_read,data_read_solar],axis=1)
# def get_solar_
def get_demand_fc_data(dt=1,d=7):    
# gets historical data from nest server
# dt: resampling period in seconds
# d:  days past, default is a week,eg 7

    now=datetime.datetime.now()
    bck=datetime.timedelta(days=1)
    start_date_fc=now-bck*(d-1)
    end_date_fc=now+bck

    start_date_demand=now-bck*d
    end_date_demand=now
    
    start_date_fc=start_date_fc.strftime('%Y-%m-%d %H:%M:%S')
    end_date_fc=end_date_fc.strftime('%Y-%m-%d %H:%M:%S')
    end_date_demand=end_date_demand.strftime('%Y-%m-%d %H:%M:%S')
    start_date_demand=start_date_demand.strftime('%Y-%m-%d %H:%M:%S')
    # print(start_date,end_date)

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata0={
    'power_bypass_ehub':'401190064',}
    metadata1={
    'outside_temperature_fc':'402190000',
    }

    df_list=[]
    df_list2=[]
    for key in list(metadata0.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata0[key]]),
                                                    startDate=start_date_demand,
                                                    endDate=end_date_demand)
            data.columns=[key]
            data=data.resample(f'{dt}S').mean().interpolate().dropna()
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    for key in list(metadata1.keys())[:]:
            try:
                data = rest.read(df_data=pd.DataFrame(columns=[metadata1[key]]),
                                                        startDate=start_date_fc,
                                                        endDate=end_date_fc)
                data.columns=[key]
                data=data.resample(f'{dt}S').mean().interpolate().dropna()
                df_list2+=[data.copy()]
            except Exception as e: 
                print(key,'did not work',e)            
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    data_read_solar=pd.concat(df_list2,axis=1)
    data_read_solar=data_read_solar.dropna()
    return pd.concat([data_read,data_read_solar],axis=1)

def get_ehub_power(start_date,dt=1,d=1):    
# gets historical data from nest server
# dt: resampling period in seconds
# d:  days past, default is a week,eg 7


    bck=datetime.timedelta(days=d)
    start_date=pd.to_datetime(start_date)
    end_date=start_date+bck
    start_date=start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')

    rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
    metadata={

    'power_bypass_ehub':'401190064',

    }

    df_list=[]
    for key in list(metadata.keys())[:]:
        try:
            data = rest.read(df_data=pd.DataFrame(columns=[metadata[key]]),
                                                    startDate=start_date,
                                                    endDate=end_date)
            data.columns=[key]
            data=data.resample(f'{dt}S').mean().interpolate().dropna()
            df_list+=[data.copy()]
        except Exception as e: 
            print(key,'did not work',e)
    data_read=pd.concat(df_list,axis=1)
    data_read=data_read.dropna()
    return data_read    

if __name__=='__main__':
    # print(get_resampled())
    print(get_ehub_power('2022-03-07'))
