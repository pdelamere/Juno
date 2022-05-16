from juno_classes import *
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import csv
import pathlib
from juno_functions import _get_files
import scipy
from spacepy import pycdf

myatan2 = np.vectorize(m.atan2)
Rj = 7.14e4

class mSWiM:
    def __init__(self):
        self.t = 0.0
        self.rhov2 = 0.0
        self.dphi = 0.0
        
        self.t, self.rhov2, self.dphi = self.read_mSWiM()
    
    def read_mSWiM(self):
        df2 = pd.read_csv('./mSWiM/mSWiM_2018.csv')    
        df2.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
        tm_df = pd.DataFrame({'year': df2['year'], 'month': df2['month'], 'day': df2['day'], 'hour': df2['hour']})
        tm = pd.to_datetime(tm_df)
        df2['tm'] = tm
        df2 = df2.set_index('tm')
        df = pd.read_csv('./mSWiM/mSWiM_2016.csv')    
        df.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
        tm_df = pd.DataFrame({'year': df['year'], 'month': df['month'], 'day': df['day'], 'hour': df['hour']})
        tm = pd.to_datetime(tm_df)
        df['tm'] = tm
        df = df.set_index('tm')
        df1 = pd.read_csv('./mSWiM/mSWiM_2017.csv')    
        df1.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
        tm_df = pd.DataFrame({'year': df1['year'], 'month': df1['month'], 'day': df1['day'], 'hour': df1['hour']})
        tm = pd.to_datetime(tm_df)
        df1['tm'] = tm
        df1 = df1.set_index('tm')
        
        df = df.append(df1)
        df = df.append(df2)
        
        rhov2 = df.rho*df.vr*df.vr

        return df.index,rhov2, df.dphi


class bc_ids:
    
    def get_mp_bc(self):
        bc_df = pd.read_csv('./wholecross5.csv')
        bc_df.columns = ['DATETIME','ID']
        bc_df = bc_df.set_index('DATETIME')
        return bc_df
    
    def get_bc_mask(self):
        self.bc_id = np.ones(len(self.t))
        id = self.bc_df['ID'][:]
        bc_t = self.bc_df.index
        #t = self.jad_tm
        self.bc_id[self.t < bc_t[0]] = 0 #no ID
        for i in range(len(bc_t)-1):
            mask = np.logical_and(bc_t[i] <= self.t,self.t < bc_t[i+1])
            if id[i] == 1:
                self.bc_id[mask] = 0
        return 
    
    def sys_3_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        
        index_array = self.t #self.jad_tm
        et_array = [spice.utc2et(i) for i in index_array.strftime('%Y-%m-%dT%H:%M:%S')]
        positions, lt = spice.spkpos('JUNO', et_array,
                                     'IAU_JUPITER', 'NONE', 'JUPITER')
        rad = np.array([])
        lat = np.array([])
        lon = np.array([])
        for vector in positions:
            r, la, lo = spice.recsph(vector)
            rad = np.append(rad, r)
            lat = np.append(lat, la*180/np.pi)
            lon = np.append(lon, lo*180/np.pi)
        
        x = np.array(positions.T[0])
        y = np.array(positions.T[1])
        z = np.array(positions.T[2])
        spice.kclear()
        
        deg2rad = np.pi/180
        a = 1.66*deg2rad
        b = 0.131
        R = np.sqrt(x**2 + y**2 + z**2)/7.14e4
        c = 1.62
        d = 7.76*deg2rad
        e = 249*deg2rad
        CentEq2 = (a*np.tanh(b*R - c) + d)*np.sin(lon*deg2rad - e)
        self.z_cent = positions.T[2]/7.14e4 - R*np.sin(CentEq2)
        self.R = R

        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        
        index_array = self.t #self.jad_tm
        et_array = [spice.utc2et(i) for i in index_array.strftime('%Y-%m-%dT%H:%M:%S')]
        positions, lt = spice.spkpos('JUNO', et_array,
                                     'JUNO_JSS', 'NONE', 'JUPITER')
        x = np.array(positions.T[0])
        y = np.array(positions.T[1])
        z = np.array(positions.T[2])
        
        return x,y,z

    def Jup_dipole(self,r):
        B0 = 417e-6/1e-9   #nTesla
        Bdp = B0/r**3 #r in units of planetary radius
        return Bdp

    def smooth(self,y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    
class MagClass(bc_ids):
    def __init__(self, timeStart, timeEnd):
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.Bx = 0
        self.By = 0
        self.Bz = 0
        self.Br = 0
        self.Btheta = 0
        self.Bphi = 0
        self.btot = 0#np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        self.bend = 0#np.zeros(len(Bx))
        self.wh_in = 0#wh_in
        self.wh_out = 0#wh_out
        self.X = 0
        self.Y = 0
        self.Z = 0
        self.R = 0#R
        self.lat = 0#lat
        self.t = 0#t
        self.z_cent = 0.0
        self.bc_id = 0.0

        self.read_mag_data()
        x,y,z = self.sys_3_data()
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()

    def read_mag_data(self):
        print('reading MAG data: ', self.timeStart, self.timeEnd)
        b = MagData(self.timeStart,self.timeEnd,'/data/juno_spacecraft/data/fgm_ss',['fgm_jno','r60s'])    
    
        bx = b.data_df['BX'][self.timeStart:self.timeEnd]
        by = b.data_df['BY'][self.timeStart:self.timeEnd]
        bz = b.data_df['BZ'][self.timeStart:self.timeEnd]
        x = b.data_df['X'][self.timeStart:self.timeEnd]
        y = b.data_df['Y'][self.timeStart:self.timeEnd]
        z = b.data_df['Z'][self.timeStart:self.timeEnd]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        self.lat = myatan2(z,r)*180/np.pi
    
        self.t = bx.index
        self.Bx = bx.to_numpy()
        self.By = by.to_numpy()
        self.Bz = bz.to_numpy()
        self.X = x.to_numpy()
        self.Y = y.to_numpy()
        self.Z = z.to_numpy()
        
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.R = self.R/Rj
        
        self.cart2sph()
        self.btot = np.sqrt(self.Br**2 + self.Btheta**2 + self.Bphi**2)
    """    
    def get_bc_mask(self):
        self.bc_id = np.ones(len(self.Bx))
        id = self.bc_df['ID'][:]
        bc_t = self.bc_df.index
        t = self.t #Bx.index
        self.bc_id[t < bc_t[0]] = 0 #no ID
        for i in range(len(bc_t)-1):
            mask = np.logical_and(bc_t[i] <= t,t < bc_t[i+1])
            if id[i] == 1:
                self.bc_id[mask] = 0
        return
    """    
    def cart2sph(self):
        XsqPlusYsq = self.X**2 + self.Y**2
        r = np.sqrt(XsqPlusYsq + self.Z**2)               # r
        myatan2 = np.vectorize(m.atan2)
        theta = myatan2(np.sqrt(XsqPlusYsq),self.Z)     # theta
        phi = myatan2(self.Y,self.X)                           # phi
        xhat = 1 #x/r
        yhat = 1 #y/r
        zhat = 1 #z/r
        self.Br = self.Bx*np.sin(theta)*np.cos(phi)*xhat + self.By*np.sin(theta)*np.sin(phi)*yhat + self.Bz*np.cos(theta)*zhat
        self.Btheta = self.Bx*np.cos(theta)*np.cos(phi)*xhat + self.By*np.cos(theta)*np.sin(phi)*yhat - self.Bz*np.sin(theta)*zhat
        self.Bphi = -self.Bx*np.sin(phi)*xhat + self.By*np.cos(phi)*yhat
        return 
        
        
class DenClass:
    def __init__(self, timeStart, timeEnd):
        self.h = 0
        self.h_sig = 0
        self.h_r = 0
        self.p = 0
        self.p_sig = 0
        self.p_r = 0
        self.t = 0
        self.timeStart = timeStart
        self.timeEnd = timeEnd

    def get_proton_density(self):
        n_df = pd.read_csv('/data/juno_spacecraft/data/jad_moments/2_protons_moments_2021July.csv')
        n_df.columns = ['UTC', 'ACCUMULATION_TIME', 'ISSUES0', 'ISSUES1', 'EV_PER_Q_RANGE0', 'EV_PER_Q_RANGE1', 'SC_POS_R', 'N_CC', 'N_SIGMA_CC']
        n_df['UTC'] = pd.to_datetime(n_df['UTC'], exact = False, format='%Y-%jT%H:%M:%S')
        n_df = n_df.set_index('UTC')
        n_dens = n_df['N_CC'][self.timeStart:self.timeEnd]
        n_sig = n_df['N_SIGMA_CC'][self.timeStart:self.timeEnd]
        n_r =  n_df['SC_POS_R'][self.timeStart:self.timeEnd]
        #wh = np.logical_not(np.isnan(n_dens))
        #n_dens = n_dens[wh]
        #n_sig = n_sig[wh]
        #n_r = n_r[wh]
        #print(n_dens[np.isnan(n_dens)])
        return n_dens,n_sig,n_r
        
    def get_heavy_density(self):
        n_df = pd.read_csv('/data/juno_spacecraft/data/jad_moments/3_heavyions_moments_2021July.csv')
        n_df.columns = ['UTC', 'ACCUMULATION_TIME', 'ISSUES0', 'ISSUES1', 'EV_PER_Q_RANGE0', 'EV_PER_Q_RANGE1', 'SC_POS_R', 'N_CC', 'N_SIGMA_CC']
        n_df['UTC'] = pd.to_datetime(n_df['UTC'], exact = False, format='%Y-%jT%H:%M:%S')
        n_df = n_df.set_index('UTC')
        n_dens = n_df['N_CC'][self.timeStart:self.timeEnd]
        n_sig = n_df['N_SIGMA_CC'][self.timeStart:self.timeEnd]
        n_r =  n_df['SC_POS_R'][self.timeStart:self.timeEnd]
        #wh = np.logical_not(np.isnan(n_dens))
        #n_dens = n_dens[wh]
        #n_sig = n_sig[wh]
        #n_r = n_r[wh]
        #print(n_dens[np.isnan(n_dens)])
        return n_dens,n_sig,n_r

    def read_density(self):
        np_den,np_sig,np_r = self.get_proton_density()
        nh_den,nh_sig,nh_r = self.get_heavy_density()

        whh = nh_sig < 100*nh_den
        whp = np_sig < 100*np_den

        np_den = np_den[whp]
        np_sig = np_sig[whp]
        np_r = np_r[whp]
    
        nh_den = nh_den[whh]
        nh_sig = nh_sig[whh]
        nh_r = nh_r[whh]
        
        if (len(np_den) < len(nh_den)):
            nh_den = nh_den[np_den.index]
            nh_sig = nh_sig[np_sig.index]
            nh_r = nh_r[np_r.index]
            tden = np_r.index
        else:
            np_den = np_den[nh_den.index]
            np_sig = np_sig[nh_sig.index]
            np_r = np_r[nh_r.index]
            tden = nh_r.index

        self.h = nh_den.to_numpy()
        self.h_sig = nh_sig.to_numpy()
        self.h_r = nh_r.to_numpy()
        self.p = np_den.to_numpy()
        self.p_sig = np_sig.to_numpy()
        self.p_r = np_r.to_numpy()
        self.t = tden.to_numpy()        
        return
        
class JadClass(bc_ids):
    def __init__(self,timeStart, timeEnd):
        self.t = 0.0
        self.jad_arr = 0.0
        self.jad_mean = 0.0
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.z_cent = 0.0
        self.R = 0.0
        self.bc_df = 0.0
        self.bc_id = 0.0

        self.read_data()
        x,y,z = self.sys_3_data()
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()

        
    def read_data(self):      
        dataFolder = pathlib.Path('/data/juno_spacecraft/data/jad')
        datFiles = _get_files(self.timeStart,self.timeEnd,'.DAT',dataFolder,'JAD_L30_LRS_ION_ANY_CNT') 
        jadeIon = JadeData(datFiles,self.timeStart,self.timeEnd)
        print('getting ion data....')
        jadeIon.getIonData()
        print('ion data retrieved...')
        #plt.figure()
        #if date in jadeIon.dataDict.keys(): #Ion spectrogram portion
        jadeIonData = jadeIon.dataDict
        jadeIonData = jadeIon.ion_df  
        
        self.jad_mean = []
        self.t = jadeIon.ion_df.index
        self.jad_arr = jadeIon.ion_df.to_numpy()
        #plt.imshow(np.transpose(jad_arr),origin='lower',aspect='auto',cmap='jet')
        #plt.show()
        sz = self.jad_arr.shape
        for i in range(sz[0]):
            self.jad_mean.append(self.jad_arr[i,:-2].mean())
            #self.jad_max.append(self.jad_arr[i,:-2].max())
            #plt.figure()
            #plt.plot(jad_tm,jad_mean)
            #plt.plot(jad_tm,jad_max)
            #plt.show()
        self.jad_mean = np.array(self.jad_mean)

    def get_jad_dist(self):
        data = self.jad_mean
        wh = np.logical_and((self.z_cent < 1), (self.z_cent > -1))
        print(wh)
        #data = data[wh]
        plt.figure()
        plt.hist(data)
        plt.show()

    #def get_mp_bc():
    #    bc_df = pd.read_csv('./wholecross5.csv')
    #    #bc_df = bc_df.drop(['NOTES'], axis=1)
    #    bc_df.columns = ['DATETIME','ID']
    #    #datetime = bc_df['DATE'][:] + ' ' + bc_df['TIME'][:]
    #    #bc_df['DATETIME'] = datetime
    #    bc_df = bc_df.set_index('DATETIME')
    #    return bc_df
        
    #def get_mp_bc(self):
    #    self.bc_df = pd.read_csv('./jno_crossings_master_fixed_v6.txt')
    #    self.bc_df = self.bc_df.drop(['NOTES'], axis=1)
    #    self.bc_df.columns = ['CASE','ORBIT','DATE', 'TIME', 'ID']
    #    datetime = self.bc_df['DATE'][:] + ' ' + self.bc_df['TIME'][:]
    #    self.bc_df['DATETIME'] = datetime
    #    self.bc_df = self.bc_df.set_index('DATETIME')
    #    return 

    #def get_bc_mask(self):
    #    self.bc_id = np.ones(len(self.jad_tm))
    #    id = self.bc_df['ID'][:]
    #    bc_t = self.bc_df.index
    #    t = self.jad_tm
    #    self.bc_id[t < bc_t[0]] = 0 #no ID
    #    for i in range(len(bc_t)-1):
    #        mask = np.logical_and(bc_t[i] <= t,t < bc_t[i+1])
    #        if id[i] == 1:
    #            self.bc_id[mask] = 0
    #    return 


    
#--------------------------------------------------------------------------------------------------
    
"""
def get_mp_bc():
    bc_df = pd.read_csv('./wholecross5.csv')
    #bc_df = bc_df.drop(['NOTES'], axis=1)
    bc_df.columns = ['DATETIME','ID']
    #datetime = bc_df['DATE'][:] + ' ' + bc_df['TIME'][:]
    #bc_df['DATETIME'] = datetime
    bc_df = bc_df.set_index('DATETIME')
    return bc_df


"""

def get_3d_plot(x,y,z,mask):
    fig = plt.figure()
    ax = Axes3D(fig)
    X = x[mask]
    Y = y[mask]
    Z = z[mask]
    #ax.scatter(x[::100]/Rj, y[::100]/Rj, z[::100]/Rj,'b')
    ax.scatter(X/Rj, Y/Rj, Z/Rj,c="r")
    ax.set_xlabel('x (Rj)')
    ax.set_ylabel('y (Rj)')
    ax.set_zlabel('z (Rj)')
    ax.plot(X/Rj, Y/Rj, 'r.', zdir='z', zs=-70.)
    ax.set_xlim3d(-150,50)
    ax.set_ylim3d(-150,70)
    ax.set_zlim3d(-70,50)
    #plt.show()

def get_B_profiles_2(b,wh):

    plt.figure()
    plt.plot(b.t[wh],b.Br[wh],label='$B_r$')
    plt.plot(b.t[wh],b.Btheta[wh],label='$B_\\theta$')
    plt.plot(b.t[wh],b.Bphi[wh],label='$B_\phi$')
    #btot = np.sqrt(b.Br**2 + b.Btheta**2 + b.Bphi**2)
    plt.plot(b.t[wh],b.btot[wh],'k',label='|B|')
    plt.plot(b.t[wh],-b.btot[wh],'k',label='|B|')
    plt.legend(loc="best")
    #ax.xaxis.set_minor_locator(MultipleLocator(5))
    #plt.axes().xaxis.set_minor_locator(MultipleLocator(5))
    plt.show()
    return

def get_B_dB_profiles_2(b,winsz,wh,whjad,j,jp,jh,orbit,std_thres,w):

    m = mSWiM()
    
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.set_size_inches((12,8))
    Br_bar = j.smooth(b.Br,winsz)
    Bphi_bar = j.smooth(b.Bphi,winsz)
    Btheta_bar = j.smooth(b.Btheta,winsz)
    btot_bar = np.sqrt(Br_bar**2 + Btheta_bar**2 + Bphi_bar**2)
        
    bpar = (Br_bar*b.Br + Btheta_bar*b.Btheta + Bphi_bar*b.Bphi)/btot_bar
    bperp2 = b.btot**2 - bpar**2

    ax[0].set_title('orbit = '+str(orbit))
    
#    ax[0].plot(b.t[wh],b.Br[wh],label='$B_r$')
#    ax[0].plot(b.t[wh],b.Btheta[wh],label='$B_\\theta$')
#    ax[0].plot(b.t[wh],b.Bphi[wh],label='$B_\phi$')
#    ax[0].plot(b.t[wh],b.btot[wh],'k',label='|B|')
#    ax[0].plot(b.t[wh],-b.btot[wh],'k',label='|B|')
#    ax[0].legend(loc="best")
#    ax[0].set_ylim([-70,70])
    ax[0].plot(b.t[wh],Br_bar[wh],label='$B_r$')
    ax[0].plot(b.t[wh],Btheta_bar[wh],label='$B_\\theta$')
    ax[0].plot(b.t[wh],Bphi_bar[wh],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    #ax[1].plot(b.t[wh],btot_bar[wh],'k',label='|B|')
    #ax[1].plot(b.t[wh],-btot_bar[wh],'k',label='|B|')
    ax[0].plot(b.t[wh],bperp2[wh],'k',label='$B_\perp^2$')
    ax[0].plot(b.t[wh],bpar[wh],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_ylim([-50,50])
    ax[1].plot(j.t[whjad],j.smooth(j.jad_mean[whjad],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    #ax[2].plot(n.t,n.h,label='heavies')
    #ax[2].plot(n.t,n.p,label='protons')
    #ax[2].legend(loc="best")
    #ax[2].set_ylabel('Density (cm$^{-3}$)')
    #ax[2].set_yscale('log')
    arr = w.data_df.to_numpy()
    arr = arr.transpose()
    vmin = 1e-14
    vmax = 1e-10
    ax[2].pcolormesh(w.t,w.freq[1:40],arr[1:40,:],norm=LogNorm(vmin=5e-15, vmax=1e-10))
    ax[2].set_yscale('log')
    ax[2].set_ylabel('freq (Hz)')
    ax[2].set_xlabel('Time')
    ax[2].get_shared_x_axes().join(ax[0], ax[1], ax[2], ax[3])
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax2.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    ax[2].set_xlim([np.min(j.t),np.max(j.t)])
    ax[3].plot(m.t,m.rhov2)
    ax[3].set_xlim([np.min(j.t),np.max(j.t)])
    ax[3].set_yscale('log')
    #ax[3].set_xlabel('Time')
    ax[3].set_ylabel('mSWiM rhov2')
    ax3 = ax[3].twinx()
    ax3.set_ylabel('dphi')
    ax3.plot(m.t,m.dphi,color='grey',linewidth=0.5)
    
    #fig.savefig('orbit'+str(orbit)+'_mSWiM.png',dpi=300)
    
    #plt.show()

    tpj = j.t[j.R == j.R.min()]
    
    wh1 = np.logical_and(np.logical_and(j.R > 10.0, j.t < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.t > tpj[0]), j.bc_id == 1)

    b.wh_in = (b.R > 1.5) & (b.t < tpj[0]) & (b.bc_id == 1)
    b.wh_out = (b.R > 1.5) & (b.t > tpj[0]) & (b.bc_id == 1)
    
    fig, ax = plt.subplots(3,1,sharex=True)
    fig.set_size_inches((9,6))
    ax[0].plot(b.R[b.wh_in],Br_bar[b.wh_in],label='$B_r$')
    ax[0].plot(b.R[b.wh_in],Btheta_bar[b.wh_in],label='$B_\\theta$')
    ax[0].plot(b.R[b.wh_in],Bphi_bar[b.wh_in],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.R[b.wh_in],bperp2[b.wh_in],'k',label='$B_\perp^2$')
    ax[0].plot(b.R[b.wh_in],bpar[b.wh_in],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('Orbit '+str(orbit) + ': in bound')
    ax[0].set_ylim([-50,50])
    ax[1].plot(j.R[wh1],j.smooth(j.jad_mean[wh1],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.R[wh1],j.z_cent[wh1], color='grey',linewidth=0.5)
    ax2.plot(j.R[wh1],np.zeros(len(j.t[wh1])),':',color='grey',linewidth=0.5)
    ax[1].set_xlim([np.min(j.R[wh1]),np.max(j.R[wh1])])
    #wh_zero = (j.z_cent[:-2]*j.z_cent[1:-1] < 0) & (j.bc_id[1:-1] == 1)
    #wh_zero = np.append(False,wh_zero)
    #wh_zero = np.append(wh_zero,False)
    jad_sm = j.smooth(j.jad_mean,10)
    jad_ts = pd.Series(j.smooth(np.log(j.jad_mean[wh1]),10),index = j.t[wh1])
    #jad_ts = pd.DataFrame(j.t[wh1],j.smooth((j.jad_mean[wh1]),10), j.R[wh1],columns=['t','mean','R'])
    #jad_ts.index = jad_ts['t']
    #jad_ts1 = jad_ts #+ pd.Timedelta(hours=10)
    #jad_ts1.index = jad_ts.index + pd.Timedelta(10, 'h')
    #merge = pd.merge_asof(jad_ts,jad_ts1,left_index=True, right_index=True,direction='nearest')
    #print(jad_js)
    #ax[2].plot(j.R[wh1],jad_ts)
    #ax[2].plot(j.R[wh1],jad_ts1)
    ax[2].plot(j.R[wh1],jad_ts.rolling('9.925H').std(),'.')
    #ax[2].plot(j.R[wh1],merge.corr(),'.')
    #ax[2].plot(j.R[wh_zero],jad_sm[wh_zero],'o')
    #ax[2].set_yscale('log')
    ax[2].set_ylabel('std (log_mean)')
    ax[2].set_xlabel('Radial distance (RJ)')
    #plt.show()

    #inbound
    #x,y,z = j.sys_3_data()
    r = j.R[wh1]
    t = j.t[wh1]


    #X = x[wh1]
    #Y = y[wh1]
    #Z = z[wh1]
    stdev_arr = jad_ts.rolling('9.925H').std()
    jad_mean = j.smooth(j.jad_mean[wh1],10)
    
    mask = (stdev_arr >= std_thres)
    mask_in = mask
    ax[1].plot(r[mask],jad_mean[mask],'r.',markersize=1.0)
    fig.savefig('orbit'+str(orbit)+'_inbound_id.png',dpi=300)
    
    #get_3d_plot(X,Y,Z,mask)    
    
    """
    bin_count, bin_edges, binnumber = scipy.stats.binned_statistic(r[mask],stdev_arr[mask],statistic='count',bins=40)
    #bin_width = (bin_edges[1] - bin_edges[0])
    #bin_centers = bin_e
    plt.figure()
    plt.hlines(bin_count,bin_edges[:-1],bin_edges[1:])
    #plt.show()

    bin_start = bin_edges[:-1]
    wstart = (bin_count > 0.0) 
    #print('disc......',bin_start[wstart])
    #print('cushion...',bin_start[np.logical_not(wstart)])

    r_arr = []
    for i in range(len(wstart)-1):
        if (wstart[i] != wstart[i+1]):
            r_arr.append(bin_start[i])
            #print(r_arr)

    for i in range(len(r_arr)):
        wh = abs(r - r_arr[i]) == np.min(abs(r-r_arr[i]))
        print('inbound...',i,t[wh],r[wh],r_arr[i],np.min(abs(r-r_arr[i])))
    """        
    
    fig, ax = plt.subplots(3,1,sharex=True)
    fig.set_size_inches( (9,6))
    ax[0].plot(b.R[b.wh_out],Br_bar[b.wh_out],label='$B_r$')
    ax[0].plot(b.R[b.wh_out],Btheta_bar[b.wh_out],label='$B_\\theta$')
    ax[0].plot(b.R[b.wh_out],Bphi_bar[b.wh_out],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.R[b.wh_out],bperp2[b.wh_out],'k',label='$B_\perp^2$')
    ax[0].plot(b.R[b.wh_out],bpar[b.wh_out],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('Orbit '+str(orbit) + ': out bound')
    ax[0].set_ylim([-50,50])
    ax[1].plot(j.R[wh2],j.smooth(j.jad_mean[wh2],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.R[wh2],j.z_cent[wh2], color='grey',linewidth=0.5)
    ax2.plot(j.R[wh2],np.zeros(len(j.t[wh2])),':',color='grey',linewidth=0.5)
    jad_ts = pd.Series(j.smooth(np.log(j.jad_mean[wh2]),10),index = j.t[wh2])
    ax[2].plot(j.R[wh2],jad_ts.rolling('9.925H').std(),'.')
    ax[2].set_ylabel('std (log_mean)')
    ax[2].set_xlabel('Radial distance (RJ)')
    
    #outbound
    #x,y,z = j.sys_3_data()
    r = j.R[wh2]
    t = j.t[wh2]
    
    #X = x[wh2]
    #Y = y[wh2]
    #Z = z[wh2]
    stdev_arr = jad_ts.rolling('9.925H').std()

    #mask = (stdev_arr < 0.5)

    jad_mean = j.smooth(j.jad_mean[wh2],10)
    
    mask = (stdev_arr >= std_thres)
    mask_out = mask
    
    ax[1].plot(r[mask],jad_mean[mask],'r.',markersize=1.0)
    fig.savefig('orbit'+str(orbit)+'_outbound_id.png',dpi=300)
    
    """
    bin_count, bin_edges, binnumber = scipy.stats.binned_statistic(r[mask],stdev_arr[mask],statistic='count',bins=40)
    plt.figure()
    plt.hlines(bin_count,bin_edges[:-1],bin_edges[1:])

    bin_start = bin_edges[1:]
    wstart = (bin_count > 0.0) 
    #print('disc......',bin_start[wstart])
    #print('cushion...',bin_start[np.logical_not(wstart)])

    r_arr = []
    for i in range(len(wstart)-1):
        if (wstart[i] != wstart[i+1]):
            r_arr.append(bin_start[i])
            #print(r_arr)

    for i in range(len(r_arr)):
        wh = abs(r - r_arr[i]) == np.min(abs(r-r_arr[i]))
        print('outbound....',i,t[wh],r[wh],r_arr[i],np.min(abs(r-r_arr[i])))
    """

    #get_3d_plot(X,Y,Z,mask)    
    
    #plt.show()
    return j.R[wh1], j.R[wh2], j.t[wh1], j.t[wh2], mask_in.to_numpy(), mask_out.to_numpy()

def get_jade_variation():
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

    df2 = pd.read_csv('./mSWiM/mSWiM_2018.csv')    
    df2.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
    tm_df = pd.DataFrame({'year': df2['year'], 'month': df2['month'], 'day': df2['day'], 'hour': df2['hour']})
    tm = pd.to_datetime(tm_df)
    df2['tm'] = tm
    df2 = df2.set_index('tm')
    df = pd.read_csv('./mSWiM/mSWiM_2016.csv')    
    df.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
    tm_df = pd.DataFrame({'year': df['year'], 'month': df['month'], 'day': df['day'], 'hour': df['hour']})
    tm = pd.to_datetime(tm_df)
    df['tm'] = tm
    df = df.set_index('tm')
    df1 = pd.read_csv('./mSWiM/mSWiM_2017.csv')    
    df1.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
    tm_df = pd.DataFrame({'year': df1['year'], 'month': df1['month'], 'day': df1['day'], 'hour': df1['hour']})
    tm = pd.to_datetime(tm_df)
    df1['tm'] = tm
    df1 = df1.set_index('tm')

    df = df.append(df1)
    df = df.append(df2)
    
    rhov2 = df.rho*df.vr*df.vr
    
    diff1 = []
    diff2 = []
    diff3 = []
    diff = []

    j_c_flx = pd.Series()
    j_c_flx_max = pd.Series()
    
    for orbit in range(5,18):
        
        timeStart = orbitsData[orbit]
        timeEnd = orbitsData[orbit+1]

        print('orbit...',orbit)
        
        Rj = 7.14e4
    
        filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
        picklefile = open(filename,'rb')
        j = pickle.load(picklefile)

        #j.jad_mean[np.isnan(j.jad_mean)] = 0.0
        tpj = j.t[j.R == j.R.min()]
        
        #mask = (j.R > 20) & (j.R < 100.0) & (j.t < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        #mask = (j.R > 30) & (j.t < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        mask = (j.R > 30) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(j.smooth(j.jad_mean[mask],10),index = j.t[mask])

        #j_c_tm.append(j.t[mask])
        j_c_flx = j_c_flx.append(jad_ts)
        print(j_c_flx)

        dt = ((j.t[1:]-j.t[:-1]).median()).seconds

        winsz = int(3600.*10./dt)
        print('dt...',dt,winsz)
        jad_min= np.array(jad_ts.rolling(window=winsz,center=True).min())
        jad_max= np.array(jad_ts.rolling(window=winsz,center=True).max())
        diff.append(np.log(jad_max)-np.log(jad_min))
        j_c_flx_max = j_c_flx_max.append(pd.Series(jad_max, index = j.t[mask]))
        
        
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(df.index, rhov2)
        ax[0].set_yscale('log')
        ax[0].set_label('orbit...'+str(orbit+1))
        ax[0].set_ylabel('rho v_r^2')
        ax[1].plot(j.t[mask],jad_ts)
        ax[1].plot(j.t[mask],jad_min,'.')
        ax[1].plot(j.t[mask],jad_max,'.')
        ax[1].set_yscale('log')
        #plt.plot(j.t[mask],np.log(jad_max)-np.log(jad_min),'o')
       
        wh = np.logical_not(np.isnan(jad_max))
        r = j.R[mask] 
        bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(r[wh],jad_max[wh],statistic='median',bins=20)
        wh = np.logical_not(np.isnan(jad_min))
        bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(r[wh],jad_min[wh],statistic='median',bins=20)
        """
        plt.figure()
        plt.hlines(bin_max,bin_edges[:-1],bin_edges[1:])
        plt.hlines(bin_min,bin_edges[:-1],bin_edges[1:],'r')
        plt.xlabel('radial distance (RJ)')
        plt.ylabel('median max/min JADE counts')
        plt.yscale('log')
        """

        #bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],j.jad_mean[mask],statistic='min',bins=20)
        #bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.t[mask],j.jad_mean[mask],statistic='max',bins=100)
        
        #bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
        #bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
        #diff = np.log(bin_max)-np.log(bin_min)
        #diff1.append(diff)

        #plt.hlines(diff,bin_edges[:-1],bin_edges[1:])
        #plt.hlines(bin_min,bin_edges[:-1],bin_edges[1:])
        
        """
        mask = (j.R > 50.0) & (j.R < 80) & (j.t < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(j.jad_mean[mask],index = j.t[mask])
        
        bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
        bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
        diff = np.log(bin_max)-np.log(bin_min)
        diff2.append(diff)

        plt.hlines(diff,bin_edges[:-1],bin_edges[1:])


        mask = (j.R > 80.0) & (j.t < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(j.jad_mean[mask],index = j.t[mask])
        
        bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
        bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
        diff = np.log(bin_max)-np.log(bin_min)
        diff3.append(diff)

        plt.hlines(diff,bin_edges[:-1],bin_edges[1:])
        """
        """
        plt.xlabel('R (RJ)')
        plt.ylabel('log max-min')
        """

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(df.index, rhov2)
    ax[0].set_yscale('log')
    ax[0].set_label('orbit...'+str(orbit+1))
    ax[0].set_ylabel('rho v_r^2')
    ax2 = ax[0].twinx()
    ax2.plot(j_c_flx_max,color='orange')
    ax2.set_yscale('log')
    #ax[1].plot(j_c_flx)
    #ax[1].plot(j_c_flx_max)
    #ax[1].set_yscale('log')

    plt.figure()
    plt.plot(df.index,rhov2)
    plt.yscale('log')
    plt.ylabel('rho v_r^2')
    ax2 = plt.twinx()
    ax2.plot(j_c_flx_max,color='orange')
    ax2.set_yscale('log')
    ax2.set_ylabel('max<JADE counts>')

    plt.figure()
    plt.plot(df.dphi)
    plt.ylabel('dphi')
    
    
    plt.show()
    
    plt.figure()
    plt.hist(np.array(diff).flatten())
        
    fig, ax = plt.subplots(3,1)
    ax[0].hist(np.array(diff1).flatten(),bins=15, alpha=0.5)
    ax[0].set_title('20 < R < 50')
    ax[0].set_xlim([1,9])
    ax[1].hist(np.array(diff2).flatten(),bins=15,alpha=0.5)
    ax[1].set_title('50 < R < 80')
    ax[1].set_xlim([1,9])
    ax[2].hist(np.array(diff3).flatten(),bins=15,alpha=0.5)
    ax[2].set_title('80 < R')
    ax[2].set_xlabel('log(max-min)')
    ax[2].set_xlim([1,9])
    plt.show()
    
def get_dn_dB_profiles(b,winsz,wh,whjad,j,n,orbit):

    fig, ax = plt.subplots(2,1,sharex=True)
    Br_bar = smooth(b.Br,winsz)
    Bphi_bar = smooth(b.Bphi,winsz)
    Btheta_bar = smooth(b.Btheta,winsz)
    btot_bar = np.sqrt(Br_bar**2 + Btheta_bar**2 + Bphi_bar**2)
        
    bpar = (Br_bar*b.Br + Btheta_bar*b.Btheta + Bphi_bar*b.Bphi)/btot_bar
    bperp2 = b.btot**2 - bpar**2

    ax[0].set_title('orbit = '+str(orbit))
    
#    ax[0].plot(b.t[wh],Br_bar[wh],label='$B_r$')
#    ax[0].plot(b.t[wh],Btheta_bar[wh],label='$B_\\theta$')
#    ax[0].plot(b.t[wh],Bphi_bar[wh],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    #ax[1].plot(b.t[wh],btot_bar[wh],'k',label='|B|')
    #ax[1].plot(b.t[wh],-btot_bar[wh],'k',label='|B|')
    bperp_ts = pd.Series(bperp2[wh],index = b.t[wh])
    ax[0].plot(bperp_ts.rolling(window=winsz).mean(),label='mean')
    ax[0].plot(b.t[wh],bperp2[wh],'k',label='$B_\perp^2$')
    #ax[0].plot(b.t[wh],bpar[wh],label='$B_\parallel$')
    ax[0].legend(loc="best")
    #ax[0].set_yscale('log')
    ax[0].set_ylim([0,70])

    j.jad_mean[np.isnan(j.jad_mean)] = 0.0
    
    #mask = np.logical_and(whjad, np.logical_not(np.isnan(j.jad_mean)))
    tpj = j.t[j.R == j.R.min()]
    
    wh1 = np.logical_and(np.logical_and(j.R > 20.0, j.t < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.t > tpj[0]), j.bc_id == 1)
    mask = np.logical_and(whjad, np.logical_and(np.logical_not(np.isnan(j.jad_mean)), j.jad_mean>0))

    #mask = np.logical_and(mask,wh1)

    mask = (j.R > 20.0) & (j.bc_id == 1) & (j.t < tpj[0]) & mask
    
    j.z_cent[np.isnan(j.z_cent)] = 0.0
    jad_ts = pd.Series(j.jad_mean[mask],index = j.t[mask])

    ax[1].plot(jad_ts.rolling(window=10,center=True).mean())
    ax[1].plot(jad_ts.rolling(window=1200,center=True).mean())

    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')

    def get_25perc(arr):
        return np.percentile(arr, 25)
    def get_75perc(arr):
        return np.percentile(arr, 75)
    
    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(j.z_cent[mask],jad_ts,statistic='median',bins=50)
    plt.figure()
    plt.hlines(bin_means,bin_edges[:-1],bin_edges[1:],'b')
    #plt.vlines(bin_means,bin_edges[:-1],bin_edges[1:])
    #plt.plot(bin_edges[:-1],bin_means)
    plt.xlabel('z_cent (RJ)')
    plt.ylabel('median <JADE counts>')
    plt.yscale('log')
    perc_25, bin_edges, binnumber = scipy.stats.binned_statistic(j.z_cent[mask],jad_ts, get_25perc ,bins=50)
    #plt.hlines(perc_25,bin_edges[:-1],bin_edges[1:],'r')
    plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_25,'r:')
    perc_75, bin_edges, binnumber = scipy.stats.binned_statistic(j.z_cent[mask],jad_ts, get_75perc ,bins=50)
    #plt.hlines(perc_75,bin_edges[:-1],bin_edges[1:],'r')
    plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_75,'r:')

    
    bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
    bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
    plt.figure()
    print(bin_means)
    plt.hlines(np.log(bin_max)-np.log(bin_min),bin_edges[:-1],bin_edges[1:])
    #plt.vlines(bin_means,bin_edges[:-1],bin_edges[1:])
    #plt.plot(bin_edges[:-1],bin_means)
    plt.xlabel('R (RJ)')
    plt.ylabel('log max-min')
    #plt.yscale('log')
    perc_25, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts, get_25perc ,bins=50)
    #plt.hlines(perc_25,bin_edges[:-1],bin_edges[1:],'r')
    #plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_25,'r:')
    perc_75, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts, get_75perc ,bins=50)
    #plt.hlines(perc_75,bin_edges[:-1],bin_edges[1:],'r')
    #plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_75,'r:')

    plt.figure()
    plt.hist(np.log(bin_max)-np.log(bin_min))

    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(b.z_cent[wh],bperp2[wh],statistic='median',bins=50)
    plt.figure()
    plt.hlines(bin_means,bin_edges[:-1],bin_edges[1:])
    #plt.vlines(bin_means,bin_edges[:-1],bin_edges[1:])
    #plt.plot(bin_edges[:-1],bin_means)
    plt.xlabel('z_cent')
    plt.ylabel('median <bperp2>')
    plt.yscale('log')
    perc_25, bin_edges, binnumber = scipy.stats.binned_statistic(b.z_cent[wh],bperp2[wh], get_25perc ,bins=50)
    #plt.hlines(perc_25,bin_edges[:-1],bin_edges[1:],'r')
    plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_25,'r:')
    perc_75, bin_edges, binnumber = scipy.stats.binned_statistic(b.z_cent[wh],bperp2[wh], get_75perc ,bins=50)
    #plt.hlines(perc_75,bin_edges[:-1],bin_edges[1:],'r')
    plt.plot(0.5*(bin_edges[1:]+bin_edges[:-1]),perc_75,'r:')
    

    
    #ax[2].plot(n.t,n.h,label='heavies')
    #ax[2].plot(n.t,n.p,label='protons')
    #ax[2].legend(loc="best")
    #ax[2].set_ylabel('Density')
    #ax[2].set_yscale('log')
    ax[1].get_shared_x_axes().join(ax[0], ax[1])
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.t,j.z_cent, color='grey',linewidth=0.5)
    ax2.plot(j.t,np.zeros(len(j.t)),':',color='grey',linewidth=0.5)
    ax[1].set_xlim([np.min(j.t),np.max(j.t)])

    
    #plt.show()
    """
    tpj = j.t[j.R == j.R.min()]
    print('jad min tm...',j.R,Rj)
    
    
    wh1 = np.logical_and(np.logical_and(j.R > 1.5, j.t < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.t > tpj[0]), j.bc_id == 1)
    print(wh1)
          
    
    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.R[b.wh_in],Br_bar[b.wh_in],label='$B_r$')
    ax[0].plot(b.R[b.wh_in],Btheta_bar[b.wh_in],label='$B_\\theta$')
    ax[0].plot(b.R[b.wh_in],Bphi_bar[b.wh_in],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.R[b.wh_in],bperp2[b.wh_in],'k',label='$B_\perp^2$')
    ax[0].plot(b.R[b.wh_in],bpar[b.wh_in],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('in bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh1],smooth(j.jad_mean[wh1],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')

    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.R[b.wh_out],Br_bar[b.wh_out],label='$B_r$')
    ax[0].plot(b.R[b.wh_out],Btheta_bar[b.wh_out],label='$B_\\theta$')
    ax[0].plot(b.R[b.wh_out],Bphi_bar[b.wh_out],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.R[b.wh_out],bperp2[b.wh_out],'k',label='$B_\perp^2$')
    ax[0].plot(b.R[b.wh_out],bpar[b.wh_out],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('out bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh2],smooth(j.jad_mean[wh2],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    """
    
    plt.show()
    return

    
    
def get_B_den_profiles_2(n,b,wh):
    #r = r/7.14e4
    fig, ax = plt.subplots(2,1,sharex=True)
    #tden = np_den.index
    ax[0].plot(b.t[wh],b.Br[wh],label='$B_r$')
    ax[0].plot(b.t[wh],b.Btheta[wh],label='$B_\\theta$')
    ax[0].plot(b.t[wh],b.Bphi[wh],label='$B_\phi$')
    #ax[0].set_xlim([np.min(tden),np.max(tden)])
    ax[0].set_ylim([-100,100])
    #btot = np.sqrt(b.Br**2 + b.Btheta**2 + b.Bphi**2)
    ax[0].plot(b.t[wh],b.btot[wh],label='|B|')
    ax[0].legend(loc="best")
    #whh = n.h_sig < 10*n.h
    #whp = n.p_sig < 10*n.p

    ax[1].plot(n.t,n.h,'.',label='heavy')
    ax[1].plot(n.t,n.p,'.',label='protons')
    ax[1].set_xlim([np.min(n.t),np.max(n.t)])
    #ax[1].set_ylim([1e-3,np.max(n.p)])
    ax[1].set_yscale('log')
    ax[1].legend()
    plt.show()

    



def plot_Juno_mag(orbit):  #use orbits 1-26 for Huscher density
    orbit = orbit - 1
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

        
#    timeStart = '2016-07-01T00:00:00'
#    timeEnd = '2016-08-01T00:00:00'

    timeStart = orbitsData[orbit]
#    timeEnd = '2017-05-17T00:00:00'
    timeEnd = orbitsData[orbit+1]

    print(timeStart,timeEnd)
    
    Rj = 7.14e4
    
    b = MagData(timeStart,timeEnd,'/data/juno_spacecraft/data/fgm_ss',['fgm_jno','r60s'])
#    b = MagData(timeStart,timeEnd,'/data/juno_spacecraft/data/pickled_mag_pos',['jno_mag_pos','v01'])    
    
    j = JadClass(timeStart, timeEnd)
    
    #j.read_data()
    #j.sys_3_data()
    #j.get_jad_dist()
    filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
    import pickle
    jad_file = open(filename, 'wb')
    pickle.dump(j, jad_file)
    jad_file.close()
    """  
    filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    j = pickle.load(picklefile)
    """
    
    #n = DenClass(timeStart,timeEnd)
    #n.read_density()
    
    #j = JadClass(t,jad_arr,jad_mean)
    
    bx = b.data_df['BX'][timeStart:timeEnd]
    by = b.data_df['BY'][timeStart:timeEnd]
    bz = b.data_df['BZ'][timeStart:timeEnd]
    x = b.data_df['X'][timeStart:timeEnd]
    y = b.data_df['Y'][timeStart:timeEnd]
    z = b.data_df['Z'][timeStart:timeEnd]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = myatan2(z,r)*180/np.pi
    
    t = bx.index
    Bx = bx.to_numpy()
    By = by.to_numpy()
    Bz = bz.to_numpy()
    X = x.to_numpy()
    Y = y.to_numpy()
    Z = z.to_numpy()

    R = np.sqrt(X**2 + Y**2 + Z**2)
    R = R/Rj
    
    Br, Btheta, Bphi = cart2sph(X,Y,Z,Bx,By,Bz)

    tpj = t[r == r.min()]
    print('t perijove...',tpj[0],r.min())

    wh1 = np.logical_and(np.logical_and(r/Rj > 1.5, t < tpj[0]), bc_id == 1)
    wh2 = np.logical_and(np.logical_and(r/Rj > 1.5, t > tpj[0]), bc_id == 1)

    #get_3d_plot(x,y,z)

    b = MagClass(Bx,By,Bz,Br,Btheta,Bphi,wh1,wh2,R,lat,t)

    bc_df = b.get_mp_bc()
    bc_id = b.get_bc_mask(bx,bc_df)
    
    #get_B_den_profiles(t,r,Br,Btheta,Bphi,wh,np_den,nh_den,np_sig,nh_sig)
    
    #oc_freq = get_corr(t,R,lat,Br,Btheta,Bphi,wh1,wh2,bend,nh_den,np_den,nh_r,np_r)
    #bend, bend_bar = get_Bend_no_den(R,lat,Br,Btheta, Bphi,wh1,wh2)    

    print(t)
    """
    cdf = pycdf.CDF('mag_data_orbit_0.cdf', '')
    cdf['time'] = bx.index.tolist()
    cdf['Bx'] = Bx
    cdf['By'] = By
    cdf['Bz'] = Bz
    cdf['Br'] = Br
    cdf['Btheta'] = Btheta
    cdf['Bphi'] = Bphi
    cdf['R'] = b.R
    cdf['lat'] = b.lat
    cdf.close()
    """
    return n,b,j

#----------------------------------------------------------------------------------------


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
        
plt.close('all')
orbit_in_arr = []
orbit_out_arr = []
r_in_arr = []
r_out_arr = []
mask_in_arr = []
mask_out_arr = []
for i in range(5,6):
    orbit = i
    print('orbit...',i)
    timeStart = orbitsData[orbit-1]
    timeEnd = orbitsData[orbit]

    b = MagClass(timeStart,timeEnd)

    """
    j = JadClass(timeStart, timeEnd)
    
    #j.read_data()
    #j.sys_3_data()
    #j.get_jad_dist()
    filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
    import pickle
    jad_file = open(filename, 'wb')
    pickle.dump(j, jad_file)
    jad_file.close()
    """
    filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    j = pickle.load(picklefile)

    filename = './wav_orbit_'+str(orbit)+'.pkl'
    picklefile = open(filename,'rb')
    w = pickle.load(picklefile)

    #n = DenClass(timeStart,timeEnd)
    #n.read_density()

    jp = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
                     instrument=['PROTONS', 'V03'])
    jh = JAD_MOM_Data(timeStart, timeEnd, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
                     instrument=['HEAVIES', 'V03'])
    
    #for i in range(24,34):

    #bc_df = get_mp_bc()
    #print(bc_df['ID'])

    #orbit = 5
    #n,b,j = plot_Juno_mag(orbit)
    wh = (b.R > 10) & (b.bc_id == 1)
    whjad = (j.R > 10) & (j.bc_id == 1)
    #wh = b.R > 10
    #get_B_profiles_2(b,wh)
    std_thres = 0.75
    r_in, r_out, t_in, t_out, mask_in, mask_out = get_B_dB_profiles_2(b,10,wh,whjad,j,jp,jh,orbit,std_thres,w)
    orbit_in_arr.append(np.ones(len(r_in[mask_in]))*i)
    r_in_arr.append(r_in[mask_in])
    orbit_out_arr.append(np.ones(len(r_out[mask_out]))*i)
    r_out_arr.append(r_out[mask_out])
   
    #plt.show()
    #get_dn_dB_profiles(b,10,wh,whjad,j,n,orbit)
    #get_jade_variation()


    #get_B_den_profiles_2(n,b,wh)

#plt.show()


plt.figure()
for i in range(len(orbit_in_arr)):
    plt.plot(r_in_arr[i],orbit_in_arr[i],'.')
    plt.xlabel('Radial Distance (RJ)')
    plt.ylabel('orbit inbound')

plt.figure()
for i in range(len(orbit_out_arr)):
    plt.plot(r_out_arr[i],orbit_out_arr[i],'.')
    plt.xlabel('Radial Distance (RJ)')
    plt.ylabel('orbit outbound')

plt.show()

#get_jade_variation()
