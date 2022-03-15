import numpy as np
#import math as m
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv
import pandas as pd

df = pd.read_csv('./mSWiM/mSWiM_2018.csv')
df.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
tm_df = pd.DataFrame({'year': df['year'], 'month': df['month'], 'day': df['day'], 'hour': df['hour']})
tm = pd.to_datetime(tm_df)
df['tm'] = tm
print(df['tm'])
df = df.set_index('tm')
print(df)

rhov2 = df.rho*df.vr*df.vr

plt.figure()
fig,ax = plt.subplots(9,1,sharex=True)
ax[0].plot(df.index,rhov2)
ax[0].set_ylabel('rho v^2')
ax[0].set_yscale('log')
ax[1].plot(df['rho'])
ax[1].set_ylabel('rho')
ax[2].plot(df['vr'])
ax[2].set_ylabel('vr')
ax[3].plot(df['vt'])
ax[3].set_ylabel('vt')
ax[4].plot(df['vn'])
ax[4].set_ylabel('vn')
ax[5].plot(df['T'])
ax[5].set_ylabel('T')
ax[6].plot(df['br'])
ax[6].set_ylabel('br')
ax[7].plot(df['bt'])
ax[7].set_ylabel('bt')
ax[8].plot(df['bn'])
ax[8].set_ylabel('bn')
ax[8].get_shared_x_axes().join(ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7], ax[8])


plt.show()
