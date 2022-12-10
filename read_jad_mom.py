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
import math
import matplotlib.colors as pltclr

myatan2 = np.vectorize(math.atan2)

def plot_jp_jh_data(b,jp,jh,j,w,ji,je,sig_max,win,orbit,maxR):

    fig, ax = plt.subplots(7,1,sharex=True)
    fig.set_size_inches((12,8))

    wh = (b.bc_id == 1)
    btot = np.sqrt(b.Br**2 + b.Btheta**2 + b.Bphi**2)
    axid = 0
    ax[axid].set_title('PJ '+str(orbit))
    ax[axid].plot(b.t[wh],b.Br[wh],label='Br')
    ax[axid].plot(b.t[wh],b.Btheta[wh],label='Btheta')
    ax[axid].plot(b.t[wh],b.Bphi[wh],label='Bphi')
    ax[axid].plot(b.t[wh],btot[wh],'k')
    ax[axid].plot(b.t[wh],-btot[wh],'k')
    ax[axid].set_ylim([-20,20])
    ax[axid].set_ylabel('B (nT)')
    ax[axid].legend(loc='best')


    wh = (jp.data_df.n_sig/jp.data_df.n < sig_max) & (jp.data_df.n > 0) & (jp.bc_id == 1) & (jp.R < maxR)
    axid = 1
    ax[axid].plot(jp.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='protons')
    ax[axid].set_yscale('log')
    ax[axid].set_ylabel('Density (cc)')
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.data_df.n > 0) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[axid].plot(jh.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0,label='heavies')
    ax0 = ax[axid].twinx()
    ax0.set_ylabel('jade mean flux')
    ax0.plot(j.t,j.smooth(j.jad_mean,10),color='grey',linewidth=0.5)
    ax0.set_yscale('log')
    ax[axid].legend(loc="best")

    wh = (jp.data_df.Temp > 0) & (jp.data_df.Temp_sig/jp.data_df.Temp < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    axid = 2
    ax[axid].plot(jp.data_df.Temp[wh].rolling(win).mean(),'.',markersize=1.0)
    wh = (jh.data_df.Temp > 0) & (jh.data_df.Temp_sig/jh.data_df.Temp < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[axid].plot(jh.data_df.Temp[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[axid].set_yscale('log')
    ax[axid].set_ylabel('Temp (eV)') 
    ax1 = ax[axid].twinx()
    ax1.set_ylabel('z_cent')
    ax1.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax1.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)

    
    """
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
    """
    
    #wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.R < maxR) & (jp.bc_id == 1)
    axid = 3
    ax[axid].plot(jp.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    ax[axid].set_ylim([-500,500])
    ax[axid].set_ylabel('$v_\phi$ (km/s)')
    wh = (jh.data_df.vphi_sig/abs(jh.data_df.vphi) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[axid].plot(jh.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    ax2 = ax[axid].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax2.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    arr = w.data_df.to_numpy()
    arr = arr.transpose()
    f = w.freq.astype(np.float64)
    vmin = 1e-14
    vmax = 1e-10
    axid = 4
    im = ax[axid].pcolormesh(w.t,f[1:40],arr[1:40,:],norm=LogNorm(vmin=5e-15, vmax=1e-10))
    ax[axid].set_yscale('log')
    ax[axid].set_ylabel('freq (Hz)')
    #plt.colorbar(im,ax=ax[axid])
 
    axid = 5
    ax[axid].pcolormesh(ji.t,ji.energy_scale,np.log(ji.jad_arr),cmap='jet')
    ax[axid].set_yscale('log')
    ax[axid].set_ylabel('JAD_i (keV/q)')
    #ax[5].set_xlabel('Time')

    axid = 6
    ax[axid].pcolormesh(je.t,je.energy_scale,np.log(je.jad_arr),cmap='jet')
    ax[axid].set_yscale('log')
    ax[axid].set_ylabel('JAD_e (keV/q)')
    ax[axid].set_xlabel('Time')
    
def plot_jp_jh_vtheta(b,jp,jh,j,sig_max,win,orbit,maxR):

    fig, ax = plt.subplots(3,1,sharex=True)
    """    
    #wh = (jp.data_df.vr_sig/abs(jp.data_df.vr) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.vr_sig/abs(jp.data_df.vr) < sig_max) & (jp.R < maxR)
    ax[0].plot(jp.data_df.vr[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[0].set_ylim([-500,500])
    ax[0].set_ylabel('vr (km/s)')
    wh = (jh.data_df.vr_sig/abs(jh.data_df.vr) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[0].plot(jh.data_df.vr[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[0].plot(jh.data_df.vr[wh],'.',markersize=1.0)
    ax1 = ax[0].twinx()
    ax1.set_ylabel('z_cent')
    ax1.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax1.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    """
        #wh = (jp.data_df.vr_sig/abs(jp.data_df.vr) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.n_sig/abs(jp.data_df.n) < sig_max) & (jp.R < maxR)
    ax[0].plot(jp.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[0].set_ylim([-500,500])
    ax[0].set_yscale('log')
    ax[0].set_ylabel('density (cm$^{-3}$)')
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[0].plot(jh.data_df.n[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[0].plot(jh.data_df.vr[wh],'.',markersize=1.0)
    ax1 = ax[0].twinx()
    ax1.set_ylabel('z_cent')
    ax1.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax1.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    wh = (jp.data_df.vtheta_sig/abs(jp.data_df.vtheta) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    ax[1].plot(jp.data_df.vtheta[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[1].set_ylim([-500,500])
    ax[1].set_ylabel('$v_\\theta$ (km/s)')
    wh = (jh.data_df.vtheta_sig/abs(jh.data_df.vtheta) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[1].plot(jh.data_df.vtheta[wh].rolling(win).mean(),'.',markersize=1.0)
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax2.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    #wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.bc_id == 1) & (jp.R < maxR)
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < sig_max) & (jp.R < maxR)
    ax[2].plot(jp.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    #ax[2].set_ylim([-500,500])
    ax[2].set_ylabel('$v_\phi$ (km/s)')
    wh = (jh.data_df.vphi_sig/abs(jh.data_df.vphi) < sig_max) & (jh.bc_id == 1) & (jh.R < maxR)
    ax[2].plot(jh.data_df.vphi[wh].rolling(win).mean(),'.',markersize=1.0)
    ax3 = ax[2].twinx()
    ax3.set_ylabel('z_cent')
    ax3.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax3.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    
    
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

    
def get_H(R):
    #R = np.linspace(6,50,100)
    H = 0.75+0.5*np.log(R/6)
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


def get_Poynting(jp,jh,b,winsz):
    fig, ax = plt.subplots(5,1,sharex=True)
    fig.set_size_inches((12,8))

    tpj = jp.t[jp.R == jp.R.min()] 
    
    d = {'Btheta': b.Btheta, 'Br': b.Br, 'Bphi': b.Bphi, 'Btot2': b.Br**2 + b.Bphi**2 + b.Btheta**2}
    btheta_df = pd.DataFrame(data = d)
    btheta_df.index = b.t
    jp.data_df['R'] = jp.R
    jp.data_df['z_cent'] = jp.z_cent
    bt = pd.concat([btheta_df,jp.data_df]).sort_index()
    bt = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    bt = bt.interpolate().reindex(jp.data_df.index)

    bth = pd.concat([btheta_df,jh.data_df]).sort_index()
    bth = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    bth = bt.interpolate().reindex(jh.data_df.index)

    Br_bar = bt.Br.rolling(winsz).mean()
    Bphi_bar = bt.Bphi.rolling(winsz).mean()
    Btheta_bar = bt.Btheta.rolling(winsz).mean()
    btot_bar = np.sqrt(Br_bar**2 + Btheta_bar**2 + Bphi_bar**2)
    btot = np.sqrt(bt.Br**2 + bt.Btheta**2 + bt.Bphi**2)
    
    bpar = (Br_bar*bt.Br + Btheta_bar*bt.Btheta + Bphi_bar*bt.Bphi)/btot_bar

    b1 = Br_bar/btot_bar
    b2 = Btheta_bar/btot_bar
    b3 = Bphi_bar/btot_bar

    bperp1_hat1 = -b3
    bperp1_hat2 = 0.0
    bperp1_hat3 = b1
    
    bperp2_hat1 = b2*bperp1_hat3 - bperp1_hat2*b3
    bperp2_hat2 = - b1*bperp1_hat3 + b3*bperp1_hat1
    bperp2_hat3 = b1*bperp1_hat2 - b2*bperp1_hat1

    bdotbperp1 = bt.Br*bperp1_hat1 + bt.Btheta*bperp1_hat2 + bt.Bphi*bperp1_hat3
    bperp1_1 = bdotbperp1*bperp1_hat1
    bperp1_2 = bdotbperp1*bperp1_hat2
    bperp1_3 = bdotbperp1*bperp1_hat3
    bperp1 = np.sqrt(bperp1_1**2 + bperp1_2**2 + bperp1_3**2)

    bdotbperp2 = bt.Br*bperp2_hat1 + bt.Btheta*bperp2_hat2 + bt.Bphi*bperp2_hat3
    bperp2_1 = bdotbperp2*bperp2_hat1
    bperp2_2 = bdotbperp2*bperp2_hat2
    bperp2_3 = bdotbperp2*bperp2_hat3
    bperp2 = np.sqrt(bperp2_1**2 + bperp2_2**2 + bperp2_3**2)
    
    bperp = np.sqrt(bt.Btot2 - bpar**2)

    vperp1 = bt.vr*bperp1_hat1 + bt.vtheta*bperp1_hat2 + bt.vphi*bperp1_hat3
    vperp2 = bt.vr*bperp2_hat1 + bt.vtheta*bperp2_hat2 + bt.vphi*bperp2_hat3
    
    vpar = (Br_bar*bt.vr + Btheta_bar*bt.vtheta + Bphi_bar*bt.vphi)/btot_bar

    va0 = (btot*1e-9/np.sqrt(bt.n*1e6*1.6e-27*np.pi*4e-7)).interpolate()    
    #eoverb = np.sqrt(Eperp1**2 + Eperp2**2)/bperp/(va0/1e3)
    Eperp = bperp*va0/1e3
    Eperp1 = -bperp1*va0/1e3
    Eperp2 = bperp2*va0/1e3

    
    #Eperp1 = -vperp2*bpar
    #Eperp2 = vperp1*bpar
    Spar = Eperp1*bperp2 - Eperp2*bperp1
    Spar = Spar*1e3*1e-18/(np.pi*4e-7) #W/m^2
    Spar = Spar*1100*1e-6/(btot_bar*1e-9)
    
    vpar0 = vpar
    vtot2 = bt.vr**2 + bt.vtheta**2 + bt.vphi**2

    
    wh = (bt.index < tpj[0]) & (bt.R > 10) & (bt.n_sig/bt.n < 10) & (bt.vr_sig/bt.vr < 10)  & (bt.vphi_sig/bt.vphi < 10)


    plt.figure()
    plt.plot(bt.index[wh],bperp1[wh],label='$\delta B_{\perp,1}$')
    plt.plot(bt.index[wh],bperp2[wh],label='$\delta B_{\perp,2}$')
    plt.ylabel('$\delta B_\perp$ (nT)')
    plt.legend(loc='best')
    plt.xlabel('time')
#    plt.show()

    print(len(bt.R[wh]), len(Spar[wh]))
    ax[0].plot(bt.R[wh],Spar[wh]*1000,'.',markersize=1.0)
    ax[0].set_ylim([-1000,200])
    ax[0].set_ylabel('Poynting flux (mW/m^2)')
    ax0 = ax[0].twinx()
    ax0.set_ylabel('z_cent')
    whz = (bt.index < tpj[0])
    ax0.plot(bt.R[whz],bt.z_cent[whz], color='grey',linewidth=0.5)
    ax0.plot(bt.R[whz],np.zeros(len(bt.R[whz])),':',color='grey',linewidth=0.5)
    ax[1].plot(bt.R[wh],vperp1[wh],'.',markersize=1.0,label='vperp1')
    ax[1].plot(bt.R[wh],vperp2[wh],'.',markersize=1.0,label='vperp2')
    ax[1].plot(bt.R[wh],va0[wh]/1e3,'.',markersize=2.0,label='va')
    #ax[1].plot(vpar,label='vpar')
    ax[1].legend(loc='best')
    #ax[1].set_ylim([-1000,1000])
    ax[1].set_ylabel('velocity')
    ax[2].plot(bt.R[wh],bperp1[wh],'.',markersize=1.0,label='|bperp1|')
    ax[2].plot(bt.R[wh],bperp2[wh],'.',markersize=1.0,label='|bperp2|')
    ax[2].legend(loc='best')
    ax[2].set_ylabel('B')
    ax[3].plot(bt.R[wh],Eperp[wh],'.',markersize=1.0)
    ax[3].plot(bt.R[wh],np.sqrt(Eperp1[wh]**2 + Eperp2[wh]**2),'.',markersize=1.0)
    ax[3].set_ylabel('Eperp')
    #ax[3].set_ylim([0,100])
    btot = np.sqrt(bt.Br**2 + bt.Btheta**2 + bt.Bphi**2)
    ax[4].plot(bt.R[wh],bt.Br[wh],label='Br')
    ax[4].plot(bt.R[wh],bt.Btheta[wh],label='Btheta')
    ax[4].plot(bt.R[wh],bt.Bphi[wh],label='Bphi')
    ax[4].plot(bt.R[wh],btot[wh],'k')
    ax[4].plot(bt.R[wh],-btot[wh],'k')
    ax[4].set_ylim([-50,50])
    ax[4].legend(loc='best')
    ax[4].set_xlabel('Radial distance (RJ)')
    plt.show()
    
    return bt, bperp1    

    
def get_Walen(jp,jh,b,winsz):
    fig, ax = plt.subplots(8,1,sharex=True)
    fig.set_size_inches((12,8))

    tpj = jp.t[jp.R == jp.R.min()] 
    
    d = {'Btheta': b.Btheta, 'Br': b.Br, 'Bphi': b.Bphi, 'Btot2': b.Br**2 + b.Bphi**2 + b.Btheta**2}
    btheta_df = pd.DataFrame(data = d)
    btheta_df.index = b.t
    jp.data_df['R'] = jp.R
    jp.data_df['z_cent'] = jp.z_cent
    bt = pd.concat([btheta_df,jp.data_df]).sort_index()
    bt = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    bt = bt.interpolate().reindex(jp.data_df.index)

    bth = pd.concat([btheta_df,jh.data_df]).sort_index()
    bth = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    bth = bt.interpolate().reindex(jh.data_df.index)

    Br_bar = bt.Br.rolling(winsz).mean()
    Bphi_bar = bt.Bphi.rolling(winsz).mean()
    Btheta_bar = bt.Btheta.rolling(winsz).mean()
    btot_bar = np.sqrt(Br_bar**2 + Btheta_bar**2 + Bphi_bar**2)
        
    bpar = (Br_bar*bt.Br + Btheta_bar*bt.Btheta + Bphi_bar*bt.Bphi)/btot_bar

    b1 = Br_bar/btot_bar
    b2 = Btheta_bar/btot_bar
    b3 = Bphi_bar/btot_bar

    bperp1_hat1 = -b3
    bperp1_hat2 = 0.0
    bperp1_hat3 = b1
    
    bperp2_hat1 = b2*bperp1_hat3 - bperp1_hat2*b3
    bperp2_hat2 = - b1*bperp1_hat3 + b3*bperp1_hat1
    bperp2_hat3 = b1*bperp1_hat2 - b2*bperp1_hat1

    bdotbperp1 = bt.Br*bperp1_hat1 + bt.Btheta*bperp1_hat2 + bt.Bphi*bperp1_hat3
    bperp1_1 = bdotbperp1*bperp1_hat1
    bperp1_2 = bdotbperp1*bperp1_hat2
    bperp1_3 = bdotbperp1*bperp1_hat3
    bperp1 = np.sqrt(bperp1_1**2 + bperp1_2**2 + bperp1_3**2)

    bdotbperp2 = bt.Br*bperp2_hat1 + bt.Btheta*bperp2_hat2 + bt.Bphi*bperp2_hat3
    bperp2_1 = bdotbperp2*bperp2_hat1
    bperp2_2 = bdotbperp2*bperp2_hat2
    bperp2_3 = bdotbperp2*bperp2_hat3
    bperp2 = np.sqrt(bperp2_1**2 + bperp2_2**2 + bperp2_3**2)
    
    bperp = np.sqrt(bt.Btot2 - bpar**2)

    """
    plt.figure()
    #plt.plot(bpar**2)
    #plt.plot(bperp**2)
    #plt.plot(bperp1)
    #plt.plot(bperp2)
    plt.plot(bperp1_1)
    plt.plot(bperp1_2)
    plt.plot(bperp1_3)
    #plt.plot(bperp2_2**2 + bperp2_2**2 + bperp2_3**2)
    plt.ylim(-100,100)
    plt.show()
    """
    
    vpar = (Br_bar*bt.vr + Btheta_bar*bt.vtheta + Bphi_bar*bt.vphi)/btot_bar
    vpar0 = vpar
    vtot2 = bt.vr**2 + bt.vtheta**2 + bt.vphi**2

    wh = (bt.index < tpj[0]) & (jp.R > 10)
    whh = (bth.index < tpj[0]) & (jh.R > 10)
    vperp = np.sqrt(vtot2[wh] - vpar[wh]**2)
    vperp = vperp.rolling(10).mean()
    vperp0 = np.sqrt(vtot2 - vpar**2)
    t = bt.index[wh].to_numpy()
    vr_sig = bt.vr_sig[wh].to_numpy()
    vtheta_sig = bt.vtheta_sig[wh].to_numpy()
    vphi_sig = bt.vphi_sig[wh].to_numpy()
    n_sig = bt.n_sig[wh].to_numpy()
    n = bt.n[wh].to_numpy()
    vr = bt.vr[wh].to_numpy()
    vtheta = bt.vtheta[wh].to_numpy()
    vphi = bt.vphi[wh].to_numpy()
    
    va = (bperp[wh]*1e-9/np.sqrt(bt.n[wh]*1e6*1.6e-27*np.pi*4e-7)).interpolate()
    va = va.rolling(10).mean()
    va0 = va
    
    whnan = np.logical_not(np.isnan(va))
    va = va[whnan].to_numpy()
    vperp = vperp[whnan].to_numpy()
    
    dva = va[1:] - va[:-1]
    dv = vperp[1:] - vperp[:-1]
    t = t[whnan]
    vr_sig = vr_sig[whnan]
    vtheta_sig = vtheta_sig[whnan]
    vphi_sig = vphi_sig[whnan]
    n_sig = n_sig[whnan]
    n = n[whnan]
    vr = vr[whnan]
    vtheta = vtheta[whnan]
    vphi = vphi[whnan]

    
    vr_sig1 = vr_sig[:-1]
    vr_sig2 = vr_sig[1:]
    vr1 = vr[:-1]
    vr2 = vr[1:]
    
    vtheta_sig1 = vtheta_sig[:-1]
    vtheta_sig2 = vtheta_sig[1:]
    vtheta1 = vtheta[:-1]
    vtheta2 = vtheta[1:]
    
    vphi_sig1 = vphi_sig[:-1]
    vphi_sig2 = vphi_sig[1:]
    vphi1 = vphi[:-1]
    vphi2 = vphi[1:]
    
    n_sig1 = n_sig[:-1]
    n_sig2 = n_sig[1:]
    n1 = n[:-1]
    n2 = n[1:]
    
    t = t[:-1]
        
    ax[0].plot(bt.index[wh],bperp1_1[wh],'.',markersize=1.0,label='bperp1_1')
    ax[0].plot(bt.index[wh],bperp1_3[wh],'.',markersize=1.0,label='bperp1_3')
    ax[0].set_ylabel('B (nT)')
    ax[0].legend(loc='best')
    ax[1].plot(bt.Br[wh],label='Br')
    ax[1].plot(bt.Btheta[wh],label='Btheta')
    ax[1].plot(bt.Bphi[wh],label='Bphi')
    ax[1].legend(loc='best')
    ax[1].set_ylabel('B (nT)')
    ax[2].plot(t,abs(dva)/1e3,':',markersize=1.0,label='dv_A')
    ax[2].plot(t,abs(dv),':',markersize=1.0,label='dv')
    ax[2].legend(loc='best')
    ax[2].set_ylim([0,2000])
    #ax[2].plot(np.roll(va/1e3,1),'.')
    ax[2].set_ylabel('dv_A, dv (km/s)')
    #ax[3].plot(t,dv,':',markersize=1.0)
    d = {'dv': dv, 'dva': dva}
    dv_df = pd.DataFrame(data = d)
    dv_df.index = t
    ax[3].plot(dv_df.dv.rolling(10).corr(dv_df.dva).rolling(100).mean(),'.',markersize=1.0)
    ax[3].set_ylabel('corr coeff')
    #ax[3].set_ylim([-1000,1000])
    #ax[3].set_ylabel('v (km/s)')
    ax[4].plot(abs(vpar0[wh].rolling(10).mean()),'.',markersize=1.0,label='vpar')
    ax[4].plot(vperp0[wh].rolling(10).mean(),'.',markersize=1.0,label='vperp')
    ax[4].plot(bt.vphi.rolling(10).mean(),label='vphi')
    ax[4].plot(bt.vtheta.rolling(10).mean(),label='vtheta')
    ax[4].plot(bt.vr.rolling(10).mean(),label='vr')
    ax[4].plot(np.sqrt(bt.vr**2 + bt.vtheta**2 + bt.vphi**2).rolling(10).mean(),label='vtot')
    ax[4].legend(loc='best')
    ax[4].set_ylabel('v (km/s)')
    ax[4].set_ylim([0,2000])
    eoverb = abs(vperp0[wh]*1e3*bpar[wh])/va0
    eoverb = eoverb.rolling(10).mean()
    va0 = bpar[wh]*1e-9/np.sqrt(np.pi*bt.n[wh]*1e6*1.67e-27)
    ax[5].plot(eoverb.rolling(10).mean(),'.',markersize=1.0)
    #ax[5].plot(va0,'.',markersize=1.0)
    ax[5].set_ylabel('E/B')
    ax[5].set_ylim([0,20])
    coverwpp=3e8/(np.sqrt(1.6e-19**2*bt.n[wh]*1e6/(1.67e-27*8.85e-12))).rolling(10).mean()/1e3
    coverwph=3e8/(np.sqrt(1.6e-19**2*bth.n[whh]*1e6/(24*1.67e-27*8.85e-12))).rolling(10).mean()/1e3
    ax[6].plot(coverwpp,'.',markersize=1.0,label='protons')
    ax[6].plot(coverwph,'.',markersize=1.0,label='heavies')
    ax[6].set_ylabel('c/wpi (km)')
    ax[6].set_yscale('log')
    
    kperp = 2*np.pi/(30*abs(bt.vphi).rolling(10).mean())
    ax[7].plot(kperp*coverwpp,'.',markersize=1.0)
    ax[7].set_ylabel('k_perp*c/wpi')
    ax[7].set_yscale('log')

    plt.figure()
    plt.plot(bt.index[wh],bperp1_1[wh],label='$\delta B_{\perp,1}$')
    plt.plot(bt.index[wh],bperp1_3[wh],label='$\delta B_{\perp,2}$')
    plt.ylabel('$\delta B_\perp$ (nT)')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.show()
    
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[1].plot(bt.index[wh],bperp1_1[wh],label='bperp1_1')
    ax[1].plot(bt.index[wh],bperp1_3[wh],label='bperp1_3')
    ax[1].set_ylabel('$\delta B_\perp$ (nT)')
    ax[1].legend(loc='best')
    ax[0].plot(bt.Br[wh],label='Br')
    ax[0].plot(bt.Btheta[wh],label='Btheta')
    ax[0].plot(bt.Bphi[wh],label='Bphi')
    ax[0].legend(loc='best')
    ax[0].set_ylabel('B (nT)')
    ax[2].plot(t,abs(dva)/1e3,':',markersize=1.0,label='dv_A')
    ax[2].plot(t,abs(dv),':',markersize=1.0,label='dv')
    ax[2].legend(loc='best')
    ax[2].set_ylim([0,2000])
    #ax[2].plot(np.roll(va/1e3,1),'.')
    ax[2].set_ylabel('dv_A, dv (km/s)')
    ax[2].set_xlabel('time')
    
    maxsig = 10
    wh = (n_sig1/n1 < maxsig) & (n_sig2/n2 < maxsig) & (vr_sig1/abs(vr1) < maxsig) & (vr_sig2/abs(vr2) < maxsig) & (vtheta_sig1/abs(vtheta1) < maxsig) & (vtheta_sig2/abs(vtheta2) < maxsig) & (vphi_sig1/abs(vphi1) < maxsig) & (vphi_sig2/abs(vphi2) < maxsig) 
    
    plt.figure()
    plt.plot(abs(dva[wh]/1e3),(dv[wh]),'.',markersize=1.0)
    plt.xlabel('dv_A')
    plt.ylabel('dv')
    plt.show()
    return bt, bperp1    

def pres_bal(jh,b):
    A = 1.0
    ansi = (1+0.5*A)
    tpj = jh.t[jh.R == jh.R.min()]
    muo = np.pi*4e-7
    d = {'btheta': b.Btheta, 'btot2': b.Br**2 + b.Bphi**2 + b.Btheta**2}
    btheta_df = pd.DataFrame(data = d)
    btheta_df.index = b.t
    bt = pd.concat([btheta_df,jh.data_df]).sort_index()
    bt = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    bt = bt.interpolate().reindex(jh.data_df.index)
    pres_p = bt.n*1e6*bt.Temp*1.6e-19/1e-9/ansi #nPa 
    pres_b = bt.btot2*1e-18/(2*muo)/1e-9 #+ 0.15  #nPa
    wh = (abs(jh.R) > 25) & (abs(jh.R) < 60) & (jh.data_df.n_sig/abs(jh.data_df.n) < 10) & (jh.data_df.index < tpj[0])
    d = {'Pp': pres_p[wh], 'Pb': pres_b[wh], 'R': jh.R[wh], 't': jh.data_df.index[wh]}
    df = pd.DataFrame(data = d)
    plt.figure()
    plt.plot(df.R,df.Pp.rolling(100).mean(),'.',markersize=2.0,label='Plasma pressure')
    plt.plot(df.R,df.Pb.rolling(100).mean(),'.',markersize=2.0,label='Mag pressure')
    plt.plot(df.R,df.Pp.rolling(100).mean()+df.Pb.rolling(100).mean(),'.',markersize=2.0,label='total')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.title('orbit: '+str(orbit))
    plt.xlabel('Radial Distance (RJ)')
    plt.ylabel('Pressure (nPa)')
    plt.show()
    return pres_p[wh], pres_b[wh], jh.R[wh], jh.data_df.index[wh]
    
def get_mag_flux_transport(jp,b):
    d = {'btheta': b.Btheta}
    #x,y,z = jp.sys_3_data()
    btheta_df = pd.DataFrame(data = d)
    btheta_df.index = b.t
    #df = jp.data_df.copy()
    bt = pd.concat([btheta_df,jp.data_df]).sort_index()
    bt = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    #print(bt.index.is_unique)
    bt = bt.interpolate().reindex(jp.data_df.index)
    wh = (abs(jp.R) > 10) & (abs(jp.z_cent) < 1) & (jp.data_df.vr_sig/abs(jp.data_df.vr) < 10) & (jp.bc_id == 1)
    Efld = -bt.vr*1e3*bt.btheta*1e-9 #V/m
    E_df = pd.Series(data = Efld, index = jp.data_df.index)
    #plt.figure()
    #plt.plot(jp.R[wh],Efld[wh],'.',markersize=1.0)
    #plt.plot(jp.R[wh],E_df[wh].rolling(100).mean(),'.',markersize=1.0)
    #plt.show()
    return Efld[wh], jp.R[wh], jp.x[wh], jp.y[wh], jp.z[wh]

def get_mdot(jh,j,orbit):
    max_sigma = 10
    
    tpj = jh.t[jh.R == jh.R.min()]
    #wh = (jp.data_df.n_sig/abs(jp.data_df.n) < max_sigma) & (jp.z_cent < 1) & (jp.z_cent >-1) & (jp.bc_id == 1) & (jp.R > 10) & (jp.t < tpj[0])
    #plt.figure()
    #plt.plot(jp.data_df.n[wh],'.',markersize=1.0)
    #plt.yscale('log')
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < max_sigma) & (jh.data_df.vr_sig/abs(jh.data_df.vr) < max_sigma) & (jh.z_cent < 1) & (jh.z_cent >-1) & (jh.bc_id == 1) & (jh.R > 10) & (jh.t < tpj[0]) & (jh.R < 150)
    #X,y,z = jh.sys_3_data()
    #Plt.figure()
    #plt.plot(jh.R[wh],jh.data_df.n[wh].to_numpy(),'.',markersize=1.0)
   
    Rj = 7.14e7 #m
    cmtom = 1e6
    H = get_H(jh.R[wh])

    mofR = 2*np.pi*jh.data_df.n[wh].to_numpy()*cmtom*1*1.67e-27*H*Rj*jh.R[wh]*Rj
    Mdot = mofR*jh.data_df.vr[wh].to_numpy()*1e3
    #plt.figure()
    #plt.plot(jh.R[wh],Mdot,'.')
    #plt.xlabel('R (RJ)')
    #plt.ylabel('Mdot')
    #plt.yscale('log')
    #plt.title('{0:.2f}'.format(np.mean(Mdot))+' kg/s')
    print('Mdot...',np.mean(Mdot))
    #plt.show()
    return Mdot,jh.R[wh],jh.x[wh],jh.y[wh],jh.z[wh]

def get_Lmom(jp,jh,b):

    max_sigma = 10
    
    tpj = jh.t[jh.R == jh.R.min()]
    wh = (jh.data_df.n_sig/abs(jh.data_df.n) < max_sigma) & (jh.data_df.vr_sig/abs(jh.data_df.vr) < max_sigma) & (jh.z_cent < 1) & (jh.z_cent >-1) & (jh.bc_id == 1) & (jh.R > 10) & (jh.t < tpj[0]) & (jh.R < 150)
    Lmom = 24*1.67e-27*jh.R[wh]*jh.data_df.vphi[wh]
    
    return Lmom, jh.R[wh],jh.x[wh],jh.y[wh],jh.z[wh]
    
def get_vphi_R(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10) & (np.abs(jp.z_cent) < 1) & (jp.bc_id == 1) & (jp.R > 10) #& (jp.t < tpj[0])
    #wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10) & (np.abs(jp.z_cent) < 4) & (jp.R > 10) #& (jp.t < tpj[0])
    #x,y,z = jp.sys_3_data()
    #plt.figure()
    #plt.plot(jp.R[wh],jp.data_df.vphi[wh],'.')
    #plt.show()
    return jp.data_df.vphi[wh], jp.R[wh], jp.x[wh], jp.y[wh], jp.z[wh]

def get_vphi_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10) & (jp.bc_id == 1) & (jp.R > 10) 
    return jp.data_df.vphi[wh], jp.z_cent[wh], jp.Req[wh]

def get_density_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.n_sig/abs(jp.data_df.n) < 10) & (jp.bc_id == 1) & (jp.R > 10) 
    return jp.data_df.n[wh], jp.z_cent[wh], jp.Req[wh]

def get_Temp_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.Temp_sig/abs(jp.data_df.Temp) < 10) & (jp.bc_id == 1) & (jp.R > 10)
    return jp.data_df.Temp[wh], jp.z_cent[wh], jp.Req[wh]

def get_entropy_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.n_sig/abs(jp.data_df.n) < 10) & (jp.data_df.Temp_sig/abs(jp.data_df.Temp) < 10) & (jp.bc_id == 1) & (jp.R > 10)
    return jp.data_df.n[wh]**(-2./3.)*jp.data_df.Temp[wh], jp.z_cent[wh], jp.Req[wh]

def get_Lmom_h_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10) & (jp.bc_id == 1) & (jp.R > 20) 
    return np.sqrt(jp.x[wh]**2 + jp.y[wh]**2)*1e3*(24*1.67e-27*np.abs(jp.data_df.vphi[wh])*1e3*jp.data_df.n[wh]*1e6),jp.z_cent[wh],jp.Req[wh]

def get_Lmom_p_z(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10) & (jp.bc_id == 1) & (jp.R > 20) 
    return np.sqrt(jp.x[wh]**2 + jp.y[wh]**2)*1e3*(1*1.67e-27*np.abs(jp.data_df.vphi[wh])*1e3*jp.data_df.n[wh]*1e6),jp.z_cent[wh],jp.Req[wh] 

def get_temp_R(jp,orbit):
    tpj = jp.t[jp.R == jp.R.min()]
    wh = (jp.data_df.Temp_sig < 1000) & (jp.data_df.Temp > 0) & (jp.bc_id == 1) & (jp.R > 5) & (jp.R < 50) & (np.abs(jp.z_cent) < 2)
    #plt.figure()
    #plt.plot(jp.R[wh],jp.data_df.vphi[wh],'.')
    #plt.show()
    return jp.data_df.Temp[wh], np.abs(jp.R[wh])    

def get_v_vs_n(jp,jh):
    plt.figure()
    whp = (jp.R > 80) & (jp.data_df.n_sig/abs(jp.data_df.n) < 10) & (jp.data_df.vphi_sig/abs(jp.data_df.vphi)< 10 ) & (jp.bc_id ==1)
    whh = (jh.R > 80) & (jh.data_df.n_sig/abs(jh.data_df.n) < 10) & (jh.data_df.vphi_sig/abs(jh.data_df.vphi)< 10 ) & (jh.bc_id ==1)  
    plt.plot(jp.data_df.n[whp],jp.data_df.vphi[whp],'.',markersize=1.0)
    plt.plot(jh.data_df.n[whh],jh.data_df.vphi[whh],'.',markersize=1.0)
    plt.xscale('log')
    plt.xlabel('Density (cc)')
    plt.ylabel('vphi (km/s)')
    plt.figure()
    plt.plot(jp.data_df.n[whp],jp.data_df.vr[whp],'.',markersize=1.0)
    plt.plot(jh.data_df.n[whh],jh.data_df.vr[whh],'.',markersize=1.0)
    plt.xscale('log')
    plt.xlabel('Density (cc)')
    plt.ylabel('vr (km/s)')
    d = {'btheta': b.Btheta}
    btheta_df = pd.DataFrame(data = d)
    btheta_df.index = b.t
    #df = jp.data_df.copy()
    bt = pd.concat([btheta_df,jp.data_df]).sort_index()
    bt = bt.loc[~bt.index.duplicated(),:]  #get rid of duplicated indices, otherwise reindex won't work
    #print(bt.index.is_unique)
    bt = bt.interpolate().reindex(jp.data_df.index)
    plt.figure()
    wh = (bt.SC_POS_R < 70) & (bt.SC_POS_R > 40) & (bt.vr_sig/abs(bt.vr) < 10)
    plt.plot(bt.vr[wh],bt.btheta[wh],'.',markersize=1.0)
    plt.xlabel('vr')
    plt.ylabel('btheta')
    print(bt.columns)
    plt.show()

def vphi_vtheta_hodo(timeStart, timeEnd, jp):
    wh = (jp.data_df.index > timeStart) & (jp.data_df.index < timeEnd) & (jp.data_df.vphi_sig/abs(jp.data_df.vphi) < 10)
    plt.figure()
    vphi = jp.data_df.vphi[wh].rolling(20).mean()
    vtheta = jp.data_df.vtheta[wh].rolling(20).mean()
    tind = np.linspace(0,255,len(vphi))
    plt.plot(vphi,vtheta)
    plt.scatter(vphi,vtheta,c = tind, ec = 'k')
    plt.xlabel('vphi')
    plt.ylabel('vtheta')
    plt.show()
    
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
phi_arr = np.empty(1,dtype=float)
phi_Efld_arr = np.empty(1,dtype=float)
phi_vphi_arr = np.empty(1,dtype=float)
Efld_arr = np.empty(1,dtype=float)
R_Efld_arr = np.empty(1,dtype=float)
z_arr = np.empty(1,dtype=float)
vphi_arr = np.empty(1,dtype=float)
R_vphi_arr = np.empty(1,dtype=float)
vphiz_arr = np.empty(1,dtype=float)
temp_arr = np.empty(1,dtype=float)
R_T_arr = np.empty(1,dtype=float)

Lmom_arr = np.empty(1,dtype=float)
R_Lmom_arr = np.empty(1,dtype=float)
phi_Lmom_arr = np.empty(1,dtype=float)

z_vphi_arr = np.empty(1,dtype=float)
zR_vphi_arr = np.empty(1,dtype=float)
theta_vphi_arr = np.empty(1,dtype=float)

orbit = 5
for i in range(orbit,orbit+20):
    orbit = i

    timeStart = orbitsData[orbit-1]
    timeEnd = orbitsData[orbit]
    print('orbit...',orbit,timeStart,timeEnd
    )    
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

    #ji = JadClass(timeStart,timeEnd,species='ION')
    filename = './jad_ion_spec_orbit_'+str(orbit)+'.pkl'
    #jad_file = open(filename, 'wb')
    #pickle.dump(ji, jad_file)
    #jad_file.close()
    picklefile = open(filename,'rb')
    ji = pickle.load(picklefile)

    ##je = JadClass(timeStart,timeEnd,species='ELC')
    #filename = './jad_elec_spec_orbit_'+str(orbit)+'.pkl'
    ##jad_file = open(filename, 'wb')
    ##pickle.dump(je, jad_file)
    ##jad_file.close()
    #picklefile = open(filename,'rb')
    #je = pickle.load(picklefile)
    
    """
    jp = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
                      instrument=['PROTONS','V03']) 
    jh = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
                      instrument=['HEAVIES','V03'])

    #print('jp.x...',jp.x)
    
    filename = './jad_protons_orbit_'+str(orbit)+'.pkl'
    jad_file = open(filename, 'wb')
    pickle.dump(jp,jad_file)
    jad_file.close()

    filename = './jad_heavies_orbit_'+str(orbit)+'.pkl'
    jad_file = open(filename, 'wb')
    pickle.dump(jh,jad_file)
    jad_file.close()
    """
    
    filename = './jad_protons_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    jp = pickle.load(picklefile)

    filename = './jad_heavies_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    jh = pickle.load(picklefile)
    
    
    filename = './wav_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    w = pickle.load(picklefile)
    
    #jp = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
    #                  instrument=['PROTONS', 'V03'])
    #jh = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
    #                  instrument=['HEAVIES', 'V03'])

    #jp.plot_jad_data(1000,20,'protons')
    #jh.plot_jad_data(1000,20,'heavies')
    #plot_jp_jh_data(b,jp,jh,j,w,ji,je,10,4,orbit,150)
    #plot_jp_jh_vtheta(b,jp,jh,j,100,10,orbit,150)
    #plt.show()
    Mdot, R,x,y,z = get_mdot(jp,j,orbit)
    Mdot_arr = np.append(Mdot_arr,Mdot)
    phi_arr = np.append(phi_arr,myatan2(y,x))
    R_arr = np.append(R_arr,R)
    #plt.yscale('log')

    Efld, R_Efld, xf, yf, zf = get_mag_flux_transport(jp,b)
    Efld_arr = np.append(Efld_arr,Efld)
    phi_Efld_arr = np.append(phi_Efld_arr,myatan2(yf,xf))
    R_Efld_arr = np.append(R_Efld_arr, R_Efld)

    #pres_p, pres_b, R_pres, t_pres = pres_bal(jh,b)

    #bt, bperp = get_Walen(jp,jh,b,100)
    #bt, bperp = get_Poynting(jp,jh,b,40)
    #get_v_vs_n(jp,jh)

    #tSrt = '2017-07-26T00:00:00'
    #tEnd = '2017-07-28T00:00:00'
    #vphi_vtheta_hodo(tSrt, tEnd, jp)
    
    vphi, R, xv, yv, zv = get_vphi_R(jp,orbit)
    vphi_arr = np.append(vphi_arr,vphi)
    phi_vphi_arr = np.append(phi_vphi_arr,myatan2(yv,xv))
    R_vphi_arr = np.append(R_vphi_arr,R)
    
    vphi, z, R = get_entropy_z(jp,orbit)
    vphiz_arr = np.append(vphiz_arr,vphi)
    z_vphi_arr = np.append(z_vphi_arr,z)
    zR_vphi_arr = np.append(zR_vphi_arr,R)
    theta_vphi_arr = np.append(theta_vphi_arr,myatan2(z,R))

    """
    vphi, z, R = get_Lmom_h_z(jh,orbit)
    vphiz_arr = np.append(vphiz_arr,vphi)
    z_vphi_arr = np.append(z_vphi_arr,z)
    zR_vphi_arr = np.append(zR_vphi_arr,R)
    theta_vphi_arr = np.append(theta_vphi_arr,myatan2(z,R))
    """

    """
    temp, R = get_temp_R(jh,orbit)
    temp_arr = np.append(temp_arr, temp)
    R_T_arr = np.append(R_T_arr, R)
    """
    
    
    """
    filename = './wav_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    w = pickle.load(picklefile)
    """


"""
ind = R_Efld_arr.argsort()
R_Efld_arr = R_Efld_arr[ind]
Efld_arr = Efld_arr[ind]


ind = R_arr.argsort()
R_arr = R_arr[ind]
vphiR_arr = vphiR_arr[ind]

ind = z_arr.argsort()
z_arr = z_arr[ind]
vphiz_arr = vphiz_arr[ind]
"""

"""
ind = R_T_arr.argsort()
R_T_arr = R_T_arr[ind]
temp_arr = temp_arr[ind]
"""
#Mdot_arr = Mdot_arr[ind]

plt.figure()
df = pd.Series(data = Efld_arr, index = R_Efld_arr)
wh = R_Efld_arr > 30.0
plt.plot(df,'.',markersize=1.0)
#plt.plot(df.rolling(10).mean(),'.',markersize=1.0,label='rolling average')
plt.plot([df.index.min(),df.index.max()],[df.mean(),df.mean()],':',label='mean')
plt.plot([df.index.min(),df.index.max()],[0,0],':')
plt.ylabel('E (Wb/m/s)')
plt.xlabel('Radial Distance (RJ)')
#plt.title('Heavies')
plt.legend(loc="best")
plt.show()
im = df.plot.hist(bins=20,logy=True,ylim=[0,1e4],alpha=0.7,label='_nolegend_')
print(im)
im.plot([df.mean(),df.mean()],[0,1e4],label='mean')
im.plot([df.mean()+df.std(),df.mean()+df.std()],[0,1e4],'r:',label='stdev')
im.plot([df.mean()-df.std(),df.mean()-df.std()],[0,1e4],'r:')
im.set_xlabel('E (Wb/m/s)')
im.set_ylabel('counts')
im.legend(loc='best')
df.describe()
#plt.show()


av=np.mean(df[wh].mean())

# Efld-----------------------------------------------

phi_Efld_arr = phi_Efld_arr + np.pi
rbins = np.linspace(0,120,60)
abins = np.linspace(0,2*np.pi,60)

counts, _, _ = np.histogram2d(phi_Efld_arr, R_Efld_arr, bins=(abins,rbins))
sums, _, _ = np.histogram2d(phi_Efld_arr, R_Efld_arr, weights=Efld_arr, bins=(abins,rbins))

av_E = (sums/counts).T
wh = np.logical_not(np.isnan(av_E))
sigma_E = av_E
sigma_E = av_E[wh].std()

phi, r = np.meshgrid(abins, rbins)
fig, ax = plt.subplots(subplot_kw = dict(projection="polar"))
pc = ax.pcolormesh(phi,r,av_E/sigma_E,cmap="seismic",vmin=-2,vmax=2)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position-57),ax.get_rmax()/2.,'Radial Distance (R$_J$)',rotation=-20,ha='center',va='center')
ax.set_title('<E>/$\sigma$: '+"{0:.4f}".format(av_E[wh].mean()/sigma_E))
ax.grid(linestyle=':')
ax.set_thetamin(-20)
ax.set_thetamax(90)
ax.set_theta_offset(np.pi)
cb = fig.colorbar(pc)
cb.set_label('<E>/$\sigma$')
#plt.show()


"""
wh = (abs(jp.R) > 10) & (abs(jp.R) < 20)
plt.figure()
plt.plot(bt.btheta[wh],bt.vr[wh],'.')
plt.plot([bt.btheta[wh].min(),bt.btheta[wh].max()],[0,0],':')
plt.xlabel('btheta')
plt.ylabel('vr')
plt.show()
"""
# Mdot-----------------------------------------------

plt.figure()
df = pd.Series(data = Mdot_arr, index = R_arr)
wh = R_arr > 20.0
plt.plot(df[wh],'.')
#plt.plot(df[wh].rolling(100).mean(),'.')
plt.ylabel('Mdot (kg/s)')
plt.xlabel('Radial Distance (RJ)')
plt.yscale('log')
#wh = R_arr > 40
av=np.mean(df[wh].rolling(100).mean())
print('Mean Mdot...',av)
plt.plot([R_arr[wh].min(),R_arr[wh].max()],[av,av],':')
plt.title('Mdot...'+"{0:.2f}".format(av)+' kg/s')
#plt.show()

phi_arr = phi_arr + np.pi 
rbins = np.linspace(0,120,60)
abins = np.linspace(0,2*np.pi,60)

counts, _, _ = np.histogram2d(phi_arr, R_arr, bins=(abins,rbins))
sums, _, _ = np.histogram2d(phi_arr, R_arr, weights=Mdot_arr, bins=(abins,rbins))

phi, r = np.meshgrid(abins, rbins)
fig, ax = plt.subplots(subplot_kw = dict(projection="polar"))
pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="magma_r",vmax=250)
ax.set_thetamin(-20)
ax.set_thetamax(90)
ax.set_theta_offset(np.pi)
#ax.set_theta_direction(1)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position-57),ax.get_rmax()/2.,'Radial Distance (R$_J$)',rotation=-20,ha='center',va='center')
ax.set_title('<Mdot>: '+"{0:.2f}".format(av)+' kg/s')
ax.grid(linestyle=':')
cb = fig.colorbar(pc)
cb.set_label('Mdot (kg/s)')
#plt.show()

# vphi-----------------------------------------------

plt.figure()
df = pd.Series(data = vphi_arr, index = R_vphi_arr)
wh = R_vphi_arr > 20.0
plt.plot(df[wh],'.')
#plt.plot(df[wh].rolling(100).mean(),'.')
plt.ylabel('vphi (km/s)')
plt.xlabel('Radial Distance (RJ)')
plt.yscale('log')
#wh = R_arr > 40
av=np.mean(df[wh].rolling(100).mean())
plt.plot([R_vphi_arr[wh].min(),R_vphi_arr[wh].max()],[av,av],':')
plt.title('vphi...'+"{0:.2f}".format(av)+' km/s')
#plt.show()

phi_vphi_arr = phi_vphi_arr + np.pi
rbins = np.linspace(0,120,60)
abins = np.linspace(0,2*np.pi,60)

counts, _, _ = np.histogram2d(phi_vphi_arr, R_vphi_arr, bins=(abins,rbins))
sums, _, _ = np.histogram2d(phi_vphi_arr, R_vphi_arr, weights=vphi_arr, bins=(abins,rbins))

phi, r = np.meshgrid(abins, rbins)
fig, ax = plt.subplots(subplot_kw = dict(projection="polar"))
pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="seismic",vmin=-500,vmax=500)
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position-57),ax.get_rmax()/2.,'Radial Distance (R$_J$)',rotation=-20,ha='center',va='center')
ax.set_title('<vphi>: '+"{0:.2f}".format(av)+' km/s')
ax.grid(linestyle=':')
ax.set_thetamin(-20)
ax.set_thetamax(90)
ax.set_theta_offset(np.pi)
cb = fig.colorbar(pc)
cb.set_label('vphi (km/s)')
#plt.show()

# vphi_z-----------------------------------------------

plt.figure()
df = pd.Series(data = np.abs(vphiz_arr), index = z_vphi_arr)
wh = zR_vphi_arr > 10.0
plt.plot(df[wh],'.')
#plt.plot(df[wh].rolling(100).mean(),'.')
plt.ylabel('vphi (km/s)')
plt.xlabel('z_cent (RJ)')
#plt.yscale('log')
#wh = R_arr > 40
av=np.mean(df[wh].rolling(100).mean())
#plt.plot([z_vphi_arr[wh].min(),z_vphi_arr[wh].max()],[av,av],':')
plt.title('vphi...'+"{0:.2f}".format(av)+' km/s')
#plt.show()

theta_vphi_arr = theta_vphi_arr 
rbins = np.linspace(0,120,120)
abins = np.linspace(-np.pi,np.pi,120)

R = np.sqrt(zR_vphi_arr**2 + z_vphi_arr**2)
counts, _, _ = np.histogram2d(theta_vphi_arr, R, bins=(abins,rbins))
sums, _, _ = np.histogram2d(theta_vphi_arr, R, weights=np.log10(np.abs(vphiz_arr)), bins=(abins,rbins))
#sums, _, _ = np.histogram2d(theta_vphi_arr, R, weights=vphiz_arr, bins=(abins,rbins))

phi, r = np.meshgrid(abins, rbins)
fig, ax = plt.subplots(subplot_kw = dict(projection="polar"))
ax.set_thetamin(45)
ax.set_thetamax(-50)
ax.set_theta_offset(-np.pi)
ax.set_theta_direction(-1)
pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="magma_r",vmin=3.5,vmax=6)
#vmin = -3.5
#vmax = 6.0
#pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="seismic",vmin=vmin,vmax=vmax)

#pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="seismic",vmin=-500,vmax=500)
#pc = ax.pcolormesh(phi,r,(sums/counts).T,cmap="magma_r",vmin=0,vmax=6000)
ax.set_title('<entropy>: '+"{0:.2f}".format(av)+' eV cm$^{-3(\gamma -1)}$')
#ax.set_title('<L>: '+"{0:.2f}".format(av)+' kg/m/s')
#ax.set_title('<density>: '+"{0:.2f}".format(av)+' cm$^{-3}$')
#ax.set_title('<v$_\phi$>: '+"{0:.2f}".format(av)+' km/s')
#ax.set_title('<Temp>: '+"{0:.2f}".format(av)+' eV')

#ax.tick_params(pad=5)
#ax.set_xlabel('Radial Distance (RJ)', rotation = 45, size=12)
ax.grid(linestyle=':')
cb = fig.colorbar(pc)
cb.set_label('log10[entropy (eV cm$^{-3(\gamma-1)}$)]')
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position+30),ax.get_rmax()/2.,'Radial Distance (R$_J$)',rotation=-45,ha='center',va='center')

#cb.set_label('log10[density (cm$^{-3}$)]')
#cb.set_label('v$_\phi$ (km/s)')
#cb.set_label('Temperature (eV)')
plt.show()



"""
df = pd.Series(data = vphiR_arr, index = R_arr)
wh = R_arr > 10.0
plt.figure()
plt.plot(df[wh],'.',markersize=1.0)
plt.plot(df[wh].rolling(200).mean(),'.',markersize=1.0)
plt.ylabel('$v_\phi$ (km/s)')
plt.xlabel('Radial Distance (RJ)')
plt.title('z_cent < 4')

df = pd.Series(data = vphiz_arr, index = z_arr)
plt.figure()
plt.plot(df,'.',markersize=1.0)
plt.plot(df.rolling(100).median(),'.',markersize=1.0)
plt.ylabel('$v_\phi$ (km/s)')
plt.xlabel('z_cent (RJ)')
plt.title('10 < R < 50')
plt.show()
#wh = R_arr > 40
#av=np.mean(df[wh].rolling(100).mean())
#print('Mean Mdot...',av)
#plt.plot([R_arr[wh].min(),R_arr[wh].max()],[av,av],':')
#plt.title('Mdot...'+"{0:.2f}".format(av))
"""

"""
df = pd.Series(data = temp_arr, index = R_T_arr)
plt.figure()
plt.plot(df,'.',markersize=1.0)
plt.plot(df.rolling(200).median(),'.',markersize=1.0)
plt.ylabel('$T$ (eV)')
plt.xlabel('R (RJ)')
plt.yscale('log')
plt.title('Orbits 5-27: z_cent < 2')

plt.show()
"""

timestart = '2017-03-25 20:00:00'
timeend = '2017-03-26 02:00:00'
#write_snippet(bt,bperp, timestart,timeend)
