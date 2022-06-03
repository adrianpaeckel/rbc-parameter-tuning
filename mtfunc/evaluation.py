def metrics(data,idx,label,col,norm,resample=True,plot=True):
    # data: pd.DataFrame with measurements and estimation
    # idx: index of evaluation slice 
    # col: names of estiamtion signals
    # label: name of mmnt signal
    # norm: normalization constant for plots and metrics
    # resample: True for resampling 1H with sum method
    inputs=np.array([label])
    inputs=np.append(inputs,col)

    data=data.loc[idx,inputs].dropna()
    if resample:
        N=len(data)
        tidx=pd.timedelta_range(start='0 days',periods=N,freq='H')
        data=data.set_index(tidx).resample('1D').sum()
        N=len(data)
    if plot:
        data.plot.bar()
    li=[]
    me={}
    for c in col:
        me['mse']=np.square(data[label]-data[c]).sum()/N/norm**2
        me['mae']=np.abs(data[label]-data[c]).sum()/N/norm
        li.append(pd.DataFrame(index=[c],data=me))
    res=pd.concat(li,axis=0)
    return res