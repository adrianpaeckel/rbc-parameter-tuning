import sys
sys.path.insert(1, '../')
import restclient
import pandas as pd

file_name='nest_data_tot.csv'
start_date='2019-10-01 00:00:00'
end_date='2021-10-30 23:59:00'


rest = restclient.client(username='paad', password='n3esT?!4ever%2021')
metadata={
# ----- EL. ENERGY -----
# SFW unit
'power_sfw':'42110000',
# DFAB unit
'power_dfab':'42190139',
# MOVE unit
'power_move':'401190130',
# SOLACE unit
'power_sol':'42160255',
#Vision and Wood
'power_vw':'42120000',
#Urban Mining and Recycling
'poewr_umr':'42150423',
#Meet2Create
'power_m2c':'42140000',
# ELM01 Ehub power mmnt
'power_ehub':'401190042',
'power_bypass_ehub':'401190064',
'power_bypass_estore':'401190086',
# ----SOLAR POWER------
# SFW unit
'power_pv_sfw_t100':'42110038',
'power_pv_sfw_t101':'42110078',
'power_pv_sfw_t102':'42110118',
# DFAB unit
'power_pv_meter_dfab':'42190053',
'power_pv_dfab_t100':'421110102' ,
# MOVE building
# 'power_pv_move_t107':'3210295',
# 'power_pv_move_t108':'3210335',
'power_pv_move':'4180039',
'power_pv_move_t100':'40200397',
'power_pv_move_t100_d':'40200422',
# SOLACE unit
'power_pv_sol_t100':'42160278',
# Battery
# current:'40200103'
# 'voltage':'40200102',
# 'soc_max':'40200104',
# 'soc_min':'40200105',
# 'power_t200':'40200263',
# power_inverter_DC:'40200265'
# 'power_bat_meter':'401190152',
# 'research_mode' : '40200011', # 1:research,0: controller
# 'global_radiation':'3200008',
# 'global_radiation_fc':'402190002'
}

# for column in list(metadata.values()):
#     print(column)


df_read = rest.read(df_data=pd.DataFrame(columns=list(metadata.values())),
                                         startDate=start_date,
                                         endDate=end_date)
# df_read.columns=list(metadata.keys())
print(df_read.columns)
df_read.columns=list(metadata.keys())
print(df_read.columns)
# df_read.timestamp=pd.to_datetime(rdata0.timestamp,unit='ns')
# df_read.set_index('timestamp',inplace=True)

df_read.to_csv('/Users/adrianpaeckelripoll/%s.csv'%file_name)


