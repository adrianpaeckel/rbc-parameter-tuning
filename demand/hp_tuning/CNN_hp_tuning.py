import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from scipy.ndimage import gaussian_filter1d
dir_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['axes.grid'] = True
hour=60
day=24*hour

dt=hour


rdata0=pd.read_csv("/Users/adrianpaeckelripoll/nest_data_tot_4.csv")
rdata0.timestamp=pd.to_datetime(rdata0.timestamp,unit='ns')
rdata0.set_index('timestamp',inplace=True)
rdata0=rdata0.resample(f'{dt}T').mean().interpolate().dropna()
rdata=rdata0
rdata=rdata.loc[:,~rdata.columns.duplicated()]
rdata.rename(columns={'global_radiation_fc':'sun_fc'},inplace=True)

def baseline_fun(data):
#     peak,_=signal.find_peaks(data.values,prominence=(20, None),)
    order=12*hour/dt
    base_idx=signal.argrelmin(data.values,order=int(order))[0]
    mask=np.empty_like(data.values)
    mask[:]=np.nan
    bs=pd.Series(mask)
    bs.iloc[base_idx]=data.iloc[base_idx]
    bs=bs.ffill()
    bs=bs.bfill()
#     bs[:]=gaussian_filter1d(bs.values,sigma=2*order)
#     bs.plot()
    return bs.mean()


STEPS=day//dt

# winter=rdata[pd.Timestamp('2021-12-01'):pd.Timestamp('2022-01-25')].copy()
# summer=rdata[pd.Timestamp('2021-07-01'):pd.Timestamp('2021-07-30')].copy()
# spring=rdata[pd.Timestamp('2021-04-01'):pd.Timestamp('2021-04-30')].copy()

data=rdata[pd.Timestamp('2021-01-01'):pd.Timestamp('2022-01-01')].copy()
pv_signals=['power_pv_sfw_t100', 'power_pv_sfw_t101','power_pv_sfw_t102',
           'power_pv_sol_t100',
            'power_pv_dfab_t100','power_pv_move','power_pv_move_t100']
loads=['power_sfw', 'power_dfab', 'power_sol','power_vw','power_umr','power_m2c','power_move']
data.power_pv_dfab_t100=-data.power_pv_dfab_t100
data['power_pv']=data[pv_signals].sum(axis=1)
data['power_load']=data[loads].sum(axis=1)
data['power_demand']=data.power_load-data.power_pv
data.loc[data.power_demand.values<0,'power_demand']=0
signals=['power_demand','outside_temperature','wind_speed','rh_outside']
data[signals]=data[signals].apply(gaussian_filter1d,sigma=2,raw=True)
shft=data.reset_index()[data.index.strftime('%A')=='Monday'].index[0]
print('Cos/sin week start on: ',data.index[shft].strftime('%A'),shft)
data=data.reset_index(drop=False)
data['time']=data.index*dt-shft*dt
data['day_sin']=np.sin(2*np.pi*data.time/day)
data['day_cos']=np.cos(2*np.pi*data.time/day)
data['week_sin']=np.sin(2*np.pi*data.time/day/7)
data['week_cos']=np.cos(2*np.pi*data.time/day/7)
data['year_sin']=np.sin(2*np.pi*data.time/day/365)
data['year_cos']=np.cos(2*np.pi*data.time/day/365)
data['days']=data.time.values//(24*hour)
data.loc[:len(data)//4*3,'set']='train'
data.loc[len(data)//4*3:len(data)//6*5,'set']='validation'
data.loc[len(data)//6*5:,'set']='test'
data=data.reset_index(drop=True)
from scipy import signal
for i in range(len(data)-STEPS):
    data.loc[i+STEPS,'bs']=baseline_fun(data.power_demand.iloc[i:i+STEPS])
data.bs=data.bs.bfill()
data['power_demand']=data.power_demand-data.bs
data.loc[data.power_demand<=0,'power_demand']=0
data.loc[:,'power_demand']=data.loc[:,'power_demand'].shift(1*day//dt)
inputs=['power_demand','outside_temperature_fc',
        'day_sin','day_cos','week_sin','week_cos']
outputs=['power_demand']
data=data[outputs+inputs+['set']]
data=data.dropna()
data=data.loc[:,~data.columns.duplicated()]
N=121
example_df=data[N*day//dt:N*day//dt+4*day//dt]
example_df=example_df.drop(columns='set')
column_indices = {name: i for i, name in enumerate(data.columns)}
train_df = data[data.set=='train'].drop(columns='set')
val_df = data[data.set=='validation'].drop(columns='set')
test_df = data[data.set=='test'].drop(columns='set')
num_features = data.shape[1]
train_mean = train_df.mean()
train_std = train_df.std()
STEPS=day//dt
MAX_EPOCHS = 1000
class WindowGenerator():
    


    def __init__(self, input_width, label_width, shift,
                train_df, val_df, test_df,example_df=None,label_columns=None,input_columns=None,stride=None):
        self.label_columns = label_columns
        self.input_columns=input_columns
        # Store the raw data.
        self.stride=stride
        self.train_df = train_df[self.input_columns]
        self.val_df = val_df[self.input_columns]
        self.test_df = test_df[self.input_columns]
        self.example_df=example_df
        
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                            enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    def plot(self, model=None, plot_col='power_demand',plot_input='power_demand', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_input]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue


            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=2)
            if model is not None:
                print(inputs.shape)
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='b', label='Predictions',
                        c='#ff7f0e', s=2)
#             plt.ylim([-1,1])

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')    
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        if self.input_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.column_indices[name]] for name in self.input_columns],
                axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])


        return inputs, labels

    def make_dataset(self, data,batch_size=32,shuffle=True):
#         print(data)
        data = np.array(data, dtype=np.float32)
#         print(data.shape)
        if self.stride is not None: st=self.stride
        else: st=1
            
            
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=st,

        shuffle=shuffle,
        batch_size=batch_size,)

        ds = ds.map(self.split_window)
#         print(ds)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
#       """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, 'example_df', None)
        if result is None:
            self._example = self.train
        else: 
            self._example=self.make_dataset(self.example_df,batch_size=3,shuffle=True)
        return next(iter(self._example))
        
def compile_and_fit(model, window, patience=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=patience,mode='min',restore_best_weights=True)
    filepath='checkpoints/%s.{epoch:02d}-{val_loss:.2f}.h5'%model.name
    checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=filepath,mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(0.0001),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history
# wKstep = WindowGenerator(input_width=24, label_width=24,train_df=train_df_norm,val_df=val_df_norm,test_df=test_df_norm, shift=STEPS,
#                      label_columns=['power_demand'])
stride=STEPS
conv_width=7*day//dt


wconv = WindowGenerator(input_width=STEPS+conv_width-1, label_width=STEPS, shift=1*STEPS,
                      train_df=train_df,val_df=val_df,test_df=test_df,example_df=example_df,
                     label_columns=outputs,input_columns=inputs,stride=stride)
print(train_mean)
print(wconv.train_df.columns)
import keras_tuner as kt

def model_builder(hp):
    hp_units_1 = hp.Int('filters', min_value=STEPS, max_value=STEPS*len(inputs), step=2)
    hp_units_2 = hp.Int('units_2', min_value=STEPS, max_value=STEPS*len(inputs), step=2)
    dense = tf.keras.Sequential([
    tf.keras.layers.Input((wconv.input_width,len(inputs))),
    tf.keras.layers.Normalization(mean=train_mean.values, variance=np.square(train_std.values)),
    tf.keras.layers.Conv1D(filters=hp_units_1,kernel_size=(conv_width,),activation='relu'),
    tf.keras.layers.Dense(units=hp_units_2, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='relu'),
        
    ])
    dense.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(0.001),metrics=[tf.losses.MeanAbsoluteError()])
    return dense

tuner = kt.Hyperband(model_builder,
                     objective='loss',
                     max_epochs=300,
                     factor=3,
                     directory='td3',
                     project_name='convhptuning')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
tuner.search(wconv.train, epochs=100,validation_data=wconv.val, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)