from juno_classes import *
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv
import pathlib
#from juno_functions import _get_files
import scipy
from spacepy import pycdf
from os import fsdecode
import os

timeStart =  '2017-02-28T22:55:48'
timeEnd = '2017-04-22T19:14:57'

plt.close('all')
orbitsData = ['2016-07-31T19:46:02',
              '2016-09-23T03:44:48',
              '2016-11-15T05:36:45',
              '2017-01-07T03:11:30',
              '2017-02-28T22:55:48',
              '2017-04-22T19:14:57',
              '2017-06-14T15:58:35',
              '2017-08-06T11:44:04',
              '2017-09-28T07:51:01',
              '2017-11-20T05:57:23',
              '2018-01-12T03:52:42',
              '2018-03-05T23:55:41',
              '2018-04-27T19:36:40',  
              '2018-06-19T17:30:40',
              '2018-08-11T15:18:43',
              '2018-10-03T10:58:52',   
              '2018-11-25T07:01:26',
              '2019-01-17T05:19:21',   
              '2019-03-11T02:48:11',   
              '2019-05-02T22:18:47',   
              '2019-06-24T18:01:57',   
              '2019-08-16T16:01:52',   
              '2019-10-08T12:52:15',   
              '2019-11-30T07:39:10',
              '2020-01-22T05:44:55',   
              '2020-03-15T03:44:40',   
              '2020-05-07T00:16:41',   
              '2020-06-28T20:24:51',   
              '2020-08-20T16:08:49',   
              '2020-10-12T14:05:43',
              '2020-12-04T11:37:23',   
              '2021-01-26T07:36:06',
              '2021-03-20 08:39:35',
              '2021-05-12 15:29:09']   

orbit = 27
for i in range(orbit,orbit+1):
    orbit = i
    timeStart = orbitsData[orbit-1]
    timeEnd = orbitsData[orbit]

    
    w = WavData(timeStart,timeEnd,'/data/juno_spacecraft/data/wav',['WAV_','_E_V01'])
    

    filename = './wav_orbit_'+str(orbit)+'.pkl'
    import pickle
    wav_file = open(filename, 'wb')
    pickle.dump(w, wav_file)
    wav_file.close()
    
    
    #filename = './wav_orbit_'+str(orbit)+'.pkl'
    #picklefile = open(filename,'rb')
    #w = pickle.load(picklefile)

    arr = w.data_df.to_numpy()
    arr = arr.transpose()
    f = w.freq.astype(np.float64)
    vmin = 1e-14
    vmax = 1e-10
    plt.figure()
    plt.pcolormesh(w.t,f[1:40],arr[1:40,:],norm=LogNorm(vmin=5e-15, vmax=1e-10))
    plt.yscale('log')
    plt.ylabel('freq (Hz)')
    plt.xlabel('Time')
    plt.show()
    
    #w.plot_wav_data(0.1)
