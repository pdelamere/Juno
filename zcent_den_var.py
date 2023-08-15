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
import pickle 
from scipy import signal

def plot_jp_jh_data(b,jp,jh,j,sig_max,win,orbit,maxR):

    fig, ax = plt.subplots(6,1,sharex=True)
    fig.set_size_inches((12,8))
    wh = (jp.data_df.n_sig/jp.data_df.n < sig_max) & (jp.data_df.n > 0) & (jp.bc_id == 1) & (jp.R < maxR)
    ax[0].set_title('Orbit '+str(orbit))
    ax[0].plot(jp.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='protons')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Density (cc)')
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.data_df.n > 0) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[0].plot(jh.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='heavies')
    ax0 = ax[0].twinx()
    ax0.set_ylabel('jade mean flux')
    ax0.plot(j.t,j.smooth(j.jad_mean,10),color='grey',linewidth=0.5)
    ax0.set_yscale('log')
    ax[0].legend(loc="best")
 
    
    #wh = (jp.data_df.vr_sig/abs(jp.data_df.vr) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.vr_sig/abs(jp.data_df.vr) < sig_max) & (jp.R < maxR)
    ax[1].plot(jp.data_df.vr[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[1].set_ylim([-500,500])
    ax[1].set_ylabel('vr (km/s)')
    wh = (jh.data_df.vr_sig/abs(jh.data_df.vr) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[1].plot(jh.data_df.vr[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[1].plot(jh.data_df.vr[wh],'.',markersize=1.0)
    ax1 = ax[1].twinx()
    ax1.set_ylabel('z_cent')
    ax1.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax1.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    wh = (jp.data_df.vtheta_sig/abs(jp.data_df.vtheta) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    ax[2].plot(jp.data_df.vtheta[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[2].set_ylim([-500,500])
    ax[2].set_ylabel('$v_\\theta$ (km/s)')
    wh = (jh.data_df.vtheta_sig/abs(jh.data_df.vtheta) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[2].plot(jh.data_df.vtheta[wh].rolling(win).mean(),'.',markersize=1.0)
    ax2 = ax[2].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax2.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)

    
    #wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.R < maxR)
    ax[3].plot(jp.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[3].set_ylim([-500,500])
    ax[3].set_ylabel('$v_\phi$ (km/s)')
    wh = (jh.data_df.vphi_sig/abs(jh.data_df.vphi) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[3].plot(jh.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    ax3 = ax[3].twinx()
    ax3.set_ylabel('z_cent')
    ax3.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax3.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    wh = (jp.data_df.Temp > 0) & (jp.data_df.Temp_sig/jp.data_df.Temp < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    ax[4].plot(jp.data_df.Temp[wh].rolling(win).mean(),'.',markersize=1.0)
    wh = (jh.data_df.Temp > 0) & (jh.data_df.Temp_sig/jh.data_df.Temp < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[4].plot(jh.data_df.Temp[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[4].set_yscale('log')
    ax[4].set_ylabel('Temp (eV)') 
    ax4 = ax[4].twinx()
    ax4.set_ylabel('z_cent')
    ax4.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax4.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)

    wh = (b.bc_id == 1)
    btot = np.sqrt(b.Br**2 + b.Btheta**2 + b.Bphi**2)
    ax[5].plot(b.t[wh],b.Br[wh],label='Br')
    ax[5].plot(b.t[wh],b.Btheta[wh],label='Btheta')
    ax[5].plot(b.t[wh],b.Bphi[wh],label='Bphi')
    ax[5].plot(b.t[wh],btot[wh],'k')
    ax[5].plot(b.t[wh],-btot[wh],'k')
    ax[5].set_ylim([-50,50])
    ax[5].legend(loc='best')


def gaussian(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(sigma**2))
    
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

    
def get_H(R):
    #R = np.linspace(6,50,100)
    H = 0.75+1.5*np.log(R/6)
    #a1 = -0.116
    #a2 = 2.14
    #a3 = -2.05
    #a4 = 0.491
    #a5 = 0.126
    #r = np.log10(R)
    #h = a1 + a2*r + a3*r**2 + a4*r**3 + a5*r**4
    #H = 10**h
    #plt.figure()
    #plt.plot(R,H,'.')
    #plt.show()
    return H

def write_snippet(b, bperp, timestart,timeend):
    filename='./mag_df.csv'
    wh = (b.index > timestart) & (b.index < timeend)
    d = {'Br': b.Br[wh], 'Btheta': b.Btheta[wh], 'Bphi': b.Bphi[wh], 'Bperp': bperp[wh], 'R': b.R[wh], 'zcent': b.z_cent[wh]}
    df = pd.DataFrame(data = d)
    df.index = b.index[wh]
    df.to_csv(filename)
    plt.figure()
    plt.plot(df.Bperp)
    plt.show()

def plot_eq_den(jh,b,orbit):
    win = 40
    sig_max = 1000
    tpj = jh.t[jh.R == jh.R.min()]
    den =  jh.data_df.n.rolling(win).mean().to_numpy()
    wh = (jh.R > 15) & (jh.R < 50) & (np.logical_not(np.isnan(den))) & (jh.t < tpj[0]) & (abs(jh.z_cent) < 5) & (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max)
    
    fig = plt.figure()
    plt.scatter(jh.z_cent[wh],den[wh], s = 5, c = jh.R[wh], cmap='jet')
    plt.colorbar(label='Radial distance (R$_J$)')
    plt.xlabel('$z_{cent}$ ($R_J$)')
    plt.ylabel('Density (cm$^{-3}$)')
    plt.title('Orbit: '+str(orbit))
    plt.yscale('log')
    bin_num = 50
    mean, bin_edges, num = scipy.stats.binned_statistic(jh.z_cent[wh],den[wh] , 'mean', bin_num)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    plt.plot(bin_centers,mean,color='black',linewidth=5.0)

    p0 = [1,0,1]
    whm = np.logical_not(np.isnan(mean))
    coeff, var_matrix = scipy.optimize.curve_fit(gaussian, bin_centers[whm], mean[whm], p0 = p0)
    normal_dist = gaussian(bin_centers, *coeff)

    print('plasma sheet crossings...')
    print(f'A:{coeff[0]}, mu:{coeff[1]}, sigma:{coeff[2]}\n')
    plt.plot(bin_centers, normal_dist,linewidth=5.0)
    bin_centers1 = bin_centers

    pickle.dump([fig,jh.z_cent[wh],den[wh], bin_centers, mean, bin_centers1, normal_dist],open('Figure8.pkl','wb'))
    
    plt.figure()
    
    #plt.plot(jh.R[wh],den[wh]*np.exp((jh.z_cent[wh]-coeff[1])**2/(2*coeff[2]**2)),'.',markersize=1.0)
    plt.plot(jh.R[wh],den[wh],'.',markersize=1.0)
    plt.yscale('log')
    plt.xlabel('Radial distance ($R_J$)')
    plt.ylabel('Density (cc)')


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    
    
def plot_eq_den_b(jh,b,orbit):
    print(jh.data_df.columns)
    win = 40
    sig_max = 100
    tpj = jh.t[jh.R == jh.R.min()]
    den =  jh.data_df.n.rolling(win).mean().to_numpy()
    wh = (jh.R > 20) & (jh.R < 90) & (np.logical_not(np.isnan(den))) & (jh.t < tpj[0]) & (abs(jh.z_cent) < 20) & (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.data_df.vtheta_sig/abs(jh.data_df.vtheta) < sig_max) & (jh.bc_id == 1)

    plt.rcParams.update({'font.size':22})
    fig, ax1 = plt.subplots(figsize=(12,8))
    
    ax2 = ax1.twinx()
    ax1.plot(jh.t[wh],den[wh],'.',color='b')
    #plt.colorbar(label='Radial distance (R$_J$)')
    #ax1.set_xlabel('$Radial distance$ ($R_J$)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Density (cm$^{-3}$)')
    ax1.set_title('Orbit: '+str(orbit))
    ax1.set_yscale('log')
    bin_num = 50
    mean, bin_edges, num = scipy.stats.binned_statistic(jh.z_cent[wh],den[wh] , 'mean', bin_num)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    #plt.plot(bin_centers,mean,color='black',linewidth=5.0)

    p0 = [1,0,1]
    whm = np.logical_not(np.isnan(mean))
    coeff, var_matrix = scipy.optimize.curve_fit(gaussian, bin_centers[whm], mean[whm], p0 = p0)
    normal_dist = gaussian(bin_centers, *coeff)

    print('plasma sheet crossings...')
    print(f'A:{coeff[0]}, mu:{coeff[1]}, sigma:{coeff[2]}\n')
    #plt.plot(bin_centers, normal_dist,linewidth=5.0)

    #plt.figure()
    
    #plt.plot(jh.R[wh],den[wh]*np.exp((jh.z_cent[wh]-coeff[1])**2/(2*coeff[2]**2)),'.',markersize=1.0)
    #plt.plot(jh.R[wh],den[wh],'.',markersize=1.0)
    #plt.yscale('log')
    #plt.xlabel('Radial distance ($R_J$)')
    #plt.ylabel('Density (cc)')

    b_df = pd.Series(b.Btheta,index = b.R)
    ax2.plot(jh.t[wh],jh.data_df.vtheta[wh].rolling(win).mean(),'.',color='r')
    wht = (jh.R > 20) & (jh.R < 90) & (jh.t < tpj[0])  
    t = pd.to_datetime(jh.t[wht]).astype(int)/10**9
    t = t-t.min()
    ax2.plot(jh.t[wht],40*np.sin(+0.31*9.925*3600+t*2*np.pi/(9.925*3600)))
    ax2.set_ylabel('$v_\\theta$ (km/s)')
    #tt2 = jh.t[wh]
    #res = fit_sin(jh.t[wh], jh.data_df.vtheta[wh].rolling(win).mean())
    #ax2.plot(tt2,res["fitfunc"](tt2))
    #ax2.plot([b.R[wh].min(),b.R[wh].max()],[0,0],':')
    #wh = (b.R > 30) & (b.R < 60) & (abs(b.z_cent) < 10) 
    #ax2.plot(b_df.index[wh],100*b_df[wh].rolling(10).std(),'.')
    #plt.show()

    #plt.figure()
    #bin_means, bin_edges, binnumber = stats.binned_statisics(jh.z_cent[wh],jh.vtheta[wh].rolling(10).s
    
    
    
def find_max_den(jp,jh):
    from scipy.signal import find_peaks
    tpj = jp.t[jp.R == jp.R.min()] 

    #wh = (jp.R < 50) & (jp.R > 20)
    #wh1 = (j.R < 50) & (j.R > 20) & (j.t < tpj[0])
    win=40
    sig_max = 1000
    maxR = 50
    minR = 15
    
    fig, ax = plt.subplots(2,1,sharex=False)
    fig.set_size_inches((12,8))
    wh = (jp.data_df.n_sig/jp.data_df.n < sig_max) & (jp.data_df.n > 0) & (jp.bc_id == 1) & (jp.R < maxR) & (jp.R > minR) & (jp.t < tpj[0]) & (abs(jp.z_cent < 5))
    ax[0].set_title('Orbit '+str(orbit))
    ax[0].plot(jp.R[wh],jp.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='protons')
    #ax[0].plot(j.R[wh1],j.smooth(j.jad_mean[wh1],win),'.',markersize=1.0,label='jad_mean')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Density (cc)')
    ax[0].set_xlabel('Radial distance ($R_J$)')
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.data_df.n > 0) & (jh.bc_id == 1) & (jh.R < maxR) & (jh.R > minR) & (jh.t < tpj[0])
    ax[0].plot(jh.R[wh],jh.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='heavies')
    ax0 = ax[0].twinx()
    ax0.set_ylabel('zcent')
    wh = (jh.t < tpj[0]) & (jh.R < maxR) & (jh.R > minR)
    ax0.plot(jh.R[wh],jh.z_cent[wh],color='grey',linewidth=0.5)
    #ax0.set_yscale('log')
    ax[0].legend(loc="best")
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.data_df.n > 0) & (jh.bc_id == 1) & (jh.R < maxR) & (jh.R > minR) & (jh.t < tpj[0])
    
    #jh.data_df['max'] = jh.data_df.iloc[find_peaks(jh.data_df.n.rolling(win).mean().values,width=15, prominence = 1e-2)[0]]['n']
    denavg = jh.data_df.n.rolling(win).mean()
    #peaks = find_peaks(denavg,width=40, height = [5e-3, 10], prominence = 5e-3)[0]
    peaks = find_peaks(denavg,width=40, height = [5e-3,10])[0]
    jh.data_df['max'] = denavg[peaks]
    wh = (jh.t < tpj[0]) & (jh.R < maxR) & (jh.R > minR) & (abs(jh.z_cent) < 5)
    ax[0].scatter(jh.R[wh],jh.data_df['max'][wh],c='r')
    peaks = np.logical_not(np.isnan(jh.data_df['max'][wh]))   
    
    peaks_df = jh.data_df['max'][wh].dropna()
    peaks = peaks_df.to_numpy()
    t = peaks_df.index
    dt = t[1:]-t[:-1]
    dt = dt.astype('timedelta64[s]')/3600
    dn = 100*(np.log(peaks[1:])-np.log(peaks[:-1]))/np.log(peaks[:-1])

    ax[1].plot(dt,dn,'.')
    ax[1].set_xlabel('$\Delta$ t (hours)')
    ax[1].set_ylabel('$\Delta$ n (%)')
    
    whzcent = np.logical_not(np.isnan(jh.data_df['max']))
    
    return dt,dn,jh.data_df['max'][whzcent],jh.z_cent[whzcent]
    
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

Mdot_arr = np.empty(1,dtype=float)
R_arr = np.empty(1,dtype=float)
Efld_arr = np.empty(1,dtype=float)
R_Efld_arr = np.empty(1,dtype=float)
z_arr = np.empty(1,dtype=float)
vphiR_arr = np.empty(1,dtype=float)
vphiz_arr = np.empty(1,dtype=float)
temp_arr = np.empty(1,dtype=float)
R_T_arr = np.empty(1,dtype=float)

dt_arr = np.empty(1,dtype=float)
dn_arr = np.empty(1,dtype=float)
zcent_arr = np.empty(1,dtype=float)

orbit = 21
for i in range(orbit,orbit+1):
#for i in range(5,26):
    orbit = i
    timeStart = orbitsData[orbit-1]
    timeEnd = orbitsData[orbit]
    
    #b = MagClass(timeStart,timeEnd)
    #j = JadClass(timeStart,timeEnd)
    
    filename = './mag_60s_orbit_'+str(orbit)+'.pkl'
    #mag_file = open(filename, 'wb')
    #pickle.dump(b, mag_file)
    #mag_file.close()
    picklefile = open(filename,'rb')
    b = pickle.load(picklefile)

    filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
    #jad_file = open(filename, 'wb')
    #pickle.dump(j, jad_file)
    #jad_file.close()
    picklefile = open(filename,'rb')
    j = pickle.load(picklefile)

        
    filename = './jad_protons_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    jp = pickle.load(picklefile)

    filename = './jad_heavies_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    jh = pickle.load(picklefile)
    
    
    #jp = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
    #                  instrument=['PROTONS', 'V03'])
    #jh = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
    #                  instrument=['HEAVIES', 'V03'])

    #jp.plot_jad_data(1000,20,'protons')
    #jh.plot_jad_data(1000,20,'heavies')
    #plot_jp_jh_data(b,jp,jh,j,10,4,orbit,150)

    dt, dn, peaks, zcent = find_max_den(jp,jh)
    print('zcent...',zcent)
    dt_arr = np.append(dt_arr, dt)
    dn_arr = np.append(dn_arr, dn)
    zcent_arr = np.append(zcent_arr,zcent)    

    plot_eq_den(jh,b,orbit)
    #plot_eq_den_b(jh,b,orbit)


plt.figure()
wh = (dt_arr > 2) & (dt_arr < 11) & (abs(dn_arr) < 100)
plt.scatter(dt_arr[wh], dn_arr[wh])
plt.xlabel('$\Delta$ t (hours)')
plt.ylabel('$\Delta$ n (%)')

hist, bin_edges = np.histogram(dn_arr[wh], bins = 20)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
p0 = [1,0,1]
coeff, var_matrix = scipy.optimize.curve_fit(gaussian, bin_centers, hist, p0 = p0)
normal_dist = gaussian(bin_centers, *coeff)

print(f'A:{coeff[0]}, mu:{coeff[1]}, sigma:{coeff[2]}\n')

fig_handle = plt.figure()
plt.hist(dn_arr[wh], bins = 20)
plt.plot(bin_centers, normal_dist)
plt.xlabel('$\Delta log(n)$ (%)')
plt.ylabel('Counts')
with open('JADE_peaks_figure.pkl','wb') as f:
    pickle.dump([fig_handle,bin_centers,normal_dist,dn_arr[wh]],f)
    f.close()

plt.figure()
plt.hist(zcent_arr,bins=20)

#plt.xlim([-200,200])

    
plt.show()


