import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from itertools import product
import os


def get_data(loc,npy=True,index_col='index',pick=False):

    if pick:
        import pickle
        with open(loc, 'rb') as file:
            return pickle.load(file)
    if npy:
        return np.load(loc,allow_pickle=True)
    else:
        return pd.read_csv(loc,index_col=index_col)

def save_data(data,name,date=True,npy=True,folder=None,verbose=1,pick=False):

    if folder is None:
        if date:
            import datetime
            now=datetime.datetime.now()
            date=now.strftime("%Y_%m_%d_%H%M")
            path=name+'_'+date
        else:
            path=name
    else:
        if not os.path.exists(folder):
            os.makedirs(folder)
        if date:
            import datetime
            now=datetime.datetime.now()
            date=now.strftime("%Y_%m_%d_%H%M")
            path=os.path.join(folder,name+'_'+date)
        else:
            path=os.path.join(folder,name)
    if verbose==1:
        print(path)
    if npy & ~pick:
        np.save(path,data)
        return 0
    elif pick:
        import pickle
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        return 0
    elif isinstance(data,np.ndarray):   
        pd.DataFrame(data).to_csv(path+'.csv',index_label='index')
        return 0
    elif isinstance(data,pd.DataFrame):
        data.to_csv(path+'.csv',index_label='index')
        return 0
    else:
        raise Exception('data is not valid type')
def save_fig(name,date=True,folder=None):
    if folder is None:
        if date:
            import datetime
            now=datetime.datetime.now()
            date=now.strftime("%Y_%m_%d_%H%M")
            path=name+'_'+date+'.png'
        else:
            path=name+'.png'
    else:
        if not os.path.exists(folder):
            os.makedirs(folder)
        if date:
            import datetime
            now=datetime.datetime.now()
            date=now.strftime("%Y_%m_%d_%H%M")
            path=os.path.join(folder,name+'_'+date+'.png')
        else:
            path=os.path.join(folder,name+'.png')
    print(path)
    plt.savefig(path)

def gmax(opt):    
    maxi = np.argmax(opt.gp.Y)
    return opt.gp.X[maxi, :], opt.gp.Y[maxi,:]


def plot3d(fun,xlim=[-1,1],ylim=[-1,1],x_steps=10,y_steps=10):
    # fun: f(x,y)->R
    x=np.linspace(xlim[0],xlim[1],x_steps)
    y=np.linspace(ylim[0],ylim[1],y_steps)
    Z=np.zeros((len(x),len(y)))

    for i,x_ in enumerate(x):
        for j,y_ in enumerate(y):
            Z[i,j]=fun(x_,y_)

    X,Y=np.meshgrid(x,y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X.transpose(), Y.transpose(), Z, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


class DemandSignal:
# signal generator, all temporal units in minutes
    def __init__(self,period=24*60,dt=1,rep=1,signal=None):
        self._hour=60//dt
        self._day=24*self._hour
        self.rep=rep
        self.period=period
        self.dt=dt
        self.signal=signal
        self.time=np.arange(self.period//self.dt)/self._hour

    def step_signal(self,x,y):
        #x: array, list time percentage values end of step y[i]
        #y: step values
        if x[-1]!=1:
            x.append(1)
        assert len(x)==len(y)
        N=len(self.time)
        p=np.zeros(N)
        t0=0
        for i,x in enumerate(x):
            p[t0:int(N*x)]=y[i]
            t0=int(N*x)
        self.signal=p
    def square_signal(self,freq,p_max,duty=0.5):
    # freq: float, cycles per day
    # p_max: float, max and min 

        p=p_max*signal.square(freq*2*np.pi/self.period*self._hour*self.time*self.dt,duty)
        self.signal=p
    @property   
    def signal(self):
        return np.tile(self._signal,self.rep)
    @signal.setter  
    def signal(self,value):
        self._signal=value
    def plot(self,rep=False):
        mpl.rcParams['figure.figsize'] = (20. ,5.0)
        if self.rep==1:
            plt.plot(self.time,self.signal,label='power load')    
        else:
            plt.plot(range(len(self.signal)),self.signal)

def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))