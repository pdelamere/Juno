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

myatan2 = np.vectorize(m.atan2)
Rj = 7.14e4

class MagClass:
    def __init__(self, Bx, By, Bz, Br, Btheta, Bphi, bc_id, wh_in, wh_out, R, lat, t):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Br = Br
        self.Btheta = Btheta
        self.Bphi = Bphi
        self.btot = np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        self.bend = np.zeros(len(Bx))
        self.bc_id = bc_id
        self.wh_in = wh_in
        self.wh_out = wh_out
        self.r = R
        self.lat = lat
        self.t = t
        self.z_cent = 0.0

        self.sys_3_data()
        
    def sys_3_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        
        index_array = self.t
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
        #self.R = R
        #temp_df = pd.DataFrame({'radial_3': rad/7.14e4, 'lon_3': lon,
        #                        'lat_3': lat, 'eq_dist': z_equator}, index=index_array)
        
        #self.q_kaw_df = pd.concat([self.q_kaw_df.sort_index(), temp_df.sort_index()], axis=1)
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
        
class JadClass:
    def __init__(self,timeStart, timeEnd):
        self.jad_tm = 0.0
        self.jad_arr = 0.0
        self.jad_mean = 0.0
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.z_cent = 0.0
        self.R = 0.0
        self.bc_df = 0.0
        self.bc_id = 0.0

        self.read_data()
        self.sys_3_data()
        self.get_mp_bc()
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
        self.jad_tm = jadeIon.ion_df.index
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

    def sys_3_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        
        index_array = self.jad_tm
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
        #temp_df = pd.DataFrame({'radial_3': rad/7.14e4, 'lon_3': lon,
        #                        'lat_3': lat, 'eq_dist': z_equator}, index=index_array)
        
        #self.q_kaw_df = pd.concat([self.q_kaw_df.sort_index(), temp_df.sort_index()], axis=1)
        return 

    def get_jad_dist(self):
        data = self.jad_mean
        wh = np.logical_and((self.z_cent < 1), (self.z_cent > -1))
        print(wh)
        #data = data[wh]
        plt.figure()
        plt.hist(data)
        plt.show()
    
    def get_mp_bc(self):
        self.bc_df = pd.read_csv('./jno_crossings_master_fixed_v6.txt')
        self.bc_df = self.bc_df.drop(['NOTES'], axis=1)
        self.bc_df.columns = ['CASE','ORBIT','DATE', 'TIME', 'ID']
        datetime = self.bc_df['DATE'][:] + ' ' + self.bc_df['TIME'][:]
        self.bc_df['DATETIME'] = datetime
        self.bc_df = self.bc_df.set_index('DATETIME')
        return 

    def get_bc_mask(self):
        self.bc_id = np.ones(len(self.jad_tm))
        id = self.bc_df['ID'][:]
        bc_t = self.bc_df.index
        t = self.jad_tm
        self.bc_id[t < bc_t[0]] = 0 #no ID
        for i in range(len(bc_t)-1):
            mask = np.logical_and(bc_t[i] <= t,t < bc_t[i+1])
            if id[i] == 'Sheath':
                self.bc_id[mask] = 0
        return 


    
#--------------------------------------------------------------------------------------------------
    
def Jup_dipole(r):
    B0 = 417e-6/1e-9   #nTesla
    Bdp = B0/r**3 #r in units of planetary radius
    return Bdp

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_mp_bc():
    bc_df = pd.read_csv('./jno_crossings_master_fixed_v6.txt')
    bc_df = bc_df.drop(['NOTES'], axis=1)
    bc_df.columns = ['CASE','ORBIT','DATE', 'TIME', 'ID']
    datetime = bc_df['DATE'][:] + ' ' + bc_df['TIME'][:]
    bc_df['DATETIME'] = datetime
    bc_df = bc_df.set_index('DATETIME')
    return bc_df

def get_bc_mask(bx,bc_df):
    bc_id = np.ones(len(bx))
    id = bc_df['ID'][:]
    bc_t = bc_df.index
    t = bx.index
    bc_id[t < bc_t[0]] = 0 #no ID
    for i in range(len(bc_t)-1):
        mask = np.logical_and(bc_t[i] <= t,t < bc_t[i+1])
        if id[i] == 'Sheath':
            bc_id[mask] = 0
    #plt.plot(t,bc_id)
    #plt.show()
    return bc_id
    
def get_3d_plot(x,y,z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[::100]/Rj, y[::100]/Rj, z[::100]/Rj)
    ax.set_xlabel('x (Rj)')
    ax.set_ylabel('y (Rj)')
    ax.set_zlabel('z (Rj)')
    ax.plot(x[::100]/Rj, y[::100]/Rj, color='r', zdir='z', zs=-70.)
    ax.set_xlim3d(-150,50)
    ax.set_ylim3d(-150,70)
    ax.set_zlim3d(-70,50)
    plt.show()

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


def get_B_dB_profiles_2(b,winsz,wh,whjad,j,n,orbit):

    fig, ax = plt.subplots(3,1,sharex=True)
    Br_bar = smooth(b.Br,winsz)
    Bphi_bar = smooth(b.Bphi,winsz)
    Btheta_bar = smooth(b.Btheta,winsz)
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
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.jad_tm[whjad],smooth(j.jad_mean[whjad],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    ax[2].plot(n.t,n.h,label='heavies')
    ax[2].plot(n.t,n.p,label='protons')
    ax[2].legend(loc="best")
    ax[2].set_ylabel('Density')
    ax[2].set_yscale('log')
    ax[2].get_shared_x_axes().join(ax[0], ax[1], ax[2])
    ax2 = ax[1].twinx()
    ax2.set_ylabel('z_cent')
    ax2.plot(j.jad_tm,j.z_cent, color='grey')
    ax2.plot(j.jad_tm,np.zeros(len(j.jad_tm)),':',color='grey')
    ax[2].set_xlim([np.min(j.jad_tm),np.max(j.jad_tm)])

    
    #plt.show()

    tpj = j.jad_tm[j.R == j.R.min()]
    
    wh1 = np.logical_and(np.logical_and(j.R > 1.5, j.jad_tm < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.jad_tm > tpj[0]), j.bc_id == 1)
    
    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.r[b.wh_in],Br_bar[b.wh_in],label='$B_r$')
    ax[0].plot(b.r[b.wh_in],Btheta_bar[b.wh_in],label='$B_\\theta$')
    ax[0].plot(b.r[b.wh_in],Bphi_bar[b.wh_in],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.r[b.wh_in],bperp2[b.wh_in],'k',label='$B_\perp^2$')
    ax[0].plot(b.r[b.wh_in],bpar[b.wh_in],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('in bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh1],smooth(j.jad_mean[wh1],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')

    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.r[b.wh_out],Br_bar[b.wh_out],label='$B_r$')
    ax[0].plot(b.r[b.wh_out],Btheta_bar[b.wh_out],label='$B_\\theta$')
    ax[0].plot(b.r[b.wh_out],Bphi_bar[b.wh_out],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.r[b.wh_out],bperp2[b.wh_out],'k',label='$B_\perp^2$')
    ax[0].plot(b.r[b.wh_out],bpar[b.wh_out],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('out bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh2],smooth(j.jad_mean[wh2],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')

    #plt.show()

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

    df = pd.read_csv('./mSWiM/mSWiM_2017.csv')    
    df.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
    tm_df = pd.DataFrame({'year': df['year'], 'month': df['month'], 'day': df['day'], 'hour': df['hour']})
    tm = pd.to_datetime(tm_df)
    df['tm'] = tm
    df = df.set_index('tm')
    df1 = pd.read_csv('./mSWiM/mSWiM_2018.csv')    
    df1.columns = ['odate','year','doy','month','day','hour','dphi','r','rho','vr','vt','vn','T','br','bt','bn']
    tm_df = pd.DataFrame({'year': df1['year'], 'month': df1['month'], 'day': df1['day'], 'hour': df1['hour']})
    tm = pd.to_datetime(tm_df)
    df1['tm'] = tm
    df1 = df1.set_index('tm')

    df = df.append(df1)
    
    rhov2 = df.rho*df.vr*df.vr
    
    diff1 = []
    diff2 = []
    diff3 = []
    diff = []

    j_c_flx = pd.Series()
    j_c_flx_max = pd.Series()
    
    for orbit in range(4,5):
        
        timeStart = orbitsData[orbit]
        timeEnd = orbitsData[orbit+1]

        print('orbit...',orbit)
        
        Rj = 7.14e4
    
        filename = './jad_mean_orbit_'+str(orbit)+'.pkl'
        picklefile = open(filename,'rb')
        j = pickle.load(picklefile)

        j.jad_mean[np.isnan(j.jad_mean)] = 0.0
        tpj = j.jad_tm[j.R == j.R.min()]
        
        #mask = (j.R > 20) & (j.R < 100.0) & (j.jad_tm < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        #mask = (j.R > 30) & (j.jad_tm < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        mask = (j.R > 30) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(smooth(j.jad_mean[mask],10),index = j.jad_tm[mask])

        #j_c_tm.append(j.jad_tm[mask])
        j_c_flx = j_c_flx.append(jad_ts)
        print(j_c_flx)

        dt = ((j.jad_tm[1:]-j.jad_tm[:-1]).median()).seconds

        winsz = int(3600.*10./dt)
        print('dt...',dt,winsz)
        jad_min= np.array(jad_ts.rolling(window=winsz,center=True).min())
        jad_max= np.array(jad_ts.rolling(window=winsz,center=True).max())
        diff.append(np.log(jad_max)-np.log(jad_min))
        j_c_flx_max = j_c_flx_max.append(pd.Series(jad_max, index = j.jad_tm[mask]))
        
        """
        fig, ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(df.index, rhov2)
        ax[0].set_yscale('log')
        ax[0].set_label('orbit...'+str(orbit+1))
        ax[0].set_ylabel('rho v_r^2')
        ax[1].plot(j.jad_tm[mask],jad_ts)
        ax[1].plot(j.jad_tm[mask],jad_min,'.')
        ax[1].plot(j.jad_tm[mask],jad_max,'.')
        ax[1].set_yscale('log')
        #plt.plot(j.jad_tm[mask],np.log(jad_max)-np.log(jad_min),'o')
        """
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
        #bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.jad_tm[mask],j.jad_mean[mask],statistic='max',bins=100)
        
        #bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
        #bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
        #diff = np.log(bin_max)-np.log(bin_min)
        #diff1.append(diff)

        #plt.hlines(diff,bin_edges[:-1],bin_edges[1:])
        #plt.hlines(bin_min,bin_edges[:-1],bin_edges[1:])
        
        """
        mask = (j.R > 50.0) & (j.R < 80) & (j.jad_tm < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(j.jad_mean[mask],index = j.jad_tm[mask])
        
        bin_min, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='min',bins=20)
        bin_max, bin_edges, binnumber = scipy.stats.binned_statistic(j.R[mask],jad_ts,statistic='max',bins=20)
        diff = np.log(bin_max)-np.log(bin_min)
        diff2.append(diff)

        plt.hlines(diff,bin_edges[:-1],bin_edges[1:])


        mask = (j.R > 80.0) & (j.jad_tm < tpj[0]) & (j.bc_id == 1) & (np.logical_not(np.isnan(j.jad_mean))) & (j.jad_mean > 0)
        jad_ts = pd.Series(j.jad_mean[mask],index = j.jad_tm[mask])
        
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
    tpj = j.jad_tm[j.R == j.R.min()]
    
    wh1 = np.logical_and(np.logical_and(j.R > 20.0, j.jad_tm < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.jad_tm > tpj[0]), j.bc_id == 1)
    mask = np.logical_and(whjad, np.logical_and(np.logical_not(np.isnan(j.jad_mean)), j.jad_mean>0))

    mask = np.logical_and(mask,wh1)
    
    j.z_cent[np.isnan(j.z_cent)] = 0.0
    jad_ts = pd.Series(j.jad_mean[mask],index = j.jad_tm[mask])

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
    ax2.plot(j.jad_tm,j.z_cent, color='grey')
    ax2.plot(j.jad_tm,np.zeros(len(j.jad_tm)),':',color='grey')
    ax[1].set_xlim([np.min(j.jad_tm),np.max(j.jad_tm)])

    
    #plt.show()
    """
    tpj = j.jad_tm[j.R == j.R.min()]
    print('jad min tm...',j.R,Rj)
    
    
    wh1 = np.logical_and(np.logical_and(j.R > 1.5, j.jad_tm < tpj[0]), j.bc_id == 1)
    wh2 = np.logical_and(np.logical_and(j.R > 1.5, j.jad_tm > tpj[0]), j.bc_id == 1)
    print(wh1)
          
    
    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.r[b.wh_in],Br_bar[b.wh_in],label='$B_r$')
    ax[0].plot(b.r[b.wh_in],Btheta_bar[b.wh_in],label='$B_\\theta$')
    ax[0].plot(b.r[b.wh_in],Bphi_bar[b.wh_in],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.r[b.wh_in],bperp2[b.wh_in],'k',label='$B_\perp^2$')
    ax[0].plot(b.r[b.wh_in],bpar[b.wh_in],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('in bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh1],smooth(j.jad_mean[wh1],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')

    fig, ax = plt.subplots(2,1,sharex=True)    
    ax[0].plot(b.r[b.wh_out],Br_bar[b.wh_out],label='$B_r$')
    ax[0].plot(b.r[b.wh_out],Btheta_bar[b.wh_out],label='$B_\\theta$')
    ax[0].plot(b.r[b.wh_out],Bphi_bar[b.wh_out],label='$B_\phi$')
    ax[0].set_ylabel('B (nT)')
    ax[0].plot(b.r[b.wh_out],bperp2[b.wh_out],'k',label='$B_\perp^2$')
    ax[0].plot(b.r[b.wh_out],bpar[b.wh_out],label='$B_\parallel$')
    ax[0].legend(loc="best")
    ax[0].set_title('out bound')
    ax[0].set_ylim([-70,70])
    ax[1].plot(j.R[wh2],smooth(j.jad_mean[wh2],10))
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Jade mean flux')
    """
    
    plt.show()


    
    
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

    
def cart2sph(x,y,z,bx,by,bz):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    myatan2 = np.vectorize(m.atan2)
    theta = myatan2(np.sqrt(XsqPlusYsq),z)     # theta
    phi = myatan2(y,x)                           # phi
    xhat = 1 #x/r
    yhat = 1 #y/r
    zhat = 1 #z/r
    Br = bx*np.sin(theta)*np.cos(phi)*xhat + by*np.sin(theta)*np.sin(phi)*yhat + bz*np.cos(theta)*zhat
    Btheta = bx*np.cos(theta)*np.cos(phi)*xhat + by*np.cos(theta)*np.sin(phi)*yhat - bz*np.sin(theta)*zhat
    Bphi = -bx*np.sin(phi)*xhat + by*np.cos(phi)*yhat
    return Br, Btheta, Bphi

def get_Bend_2(n,b):
    fig, ax = plt.subplots(3)
    #bend = np.zeros(len(b.Bx))
    b.bend = (b.Bphi/abs(b.Bphi))*(b.Br/abs(b.Br))*myatan2(np.abs(b.Bphi),np.abs(b.Br))
    bend_bar = smooth(b.bend*180/np.pi,10)
    ax[0].plot(b.lat[b.wh_in],b.bend[b.wh_in]*180/np.pi,'.')
    ax[0].plot(b.lat[b.wh_in],bend_bar[b.wh_in],'r')
    ax[0].set_xlabel('lat')
    ax[0].set_ylabel('Bend angle (deg)')
    ax[0].set_title('inbound')
    ax[1].plot(b.r[b.wh_in],b.bend[b.wh_in]*180/np.pi,'.')
    ax[1].plot(b.r[b.wh_in],bend_bar[b.wh_in],'r')
    ax[1].set_xlabel('R (RJ)')
    ax[1].set_ylabel('Bend angle (deg)')
    ax[1].set_xlim([0,np.max(b.r[b.wh_in])])
    ax[1].plot([b.r[b.wh_in].min(),b.r[b.wh_in].max()],[0,0])
    whh = n.h_sig < 50*n.h
    whp = n.p_sig < 50*n.p
    ax[2].plot(n.h_r[whh],n.h[whh],'.')
    ax[2].plot(n.p_r[whp],n.p[whp],'.')
    ax[2].set_xlim([0,np.max(b.r[b.wh_in])])
    ax[2].get_shared_x_axes().join(ax[1], ax[2])
    ax[2].set_yscale('log')
    ax[2].set_xlabel('R (RJ)')
    ax[2].set_ylabel('Heavy density (cc)')
    plt.show()

    fig, ax = plt.subplots(2)
    #bend = Bphi*Br/(abs(Bphi)*np.abs(Br))*myatan2(np.abs(Bphi),np.abs(Br))
    #bend_bar = smooth(bend*180/np.pi,60)
    ax[0].plot(b.lat[b.wh_out],b.bend[b.wh_out]*180/np.pi,'.')
    ax[0].plot(b.lat[b.wh_out],bend_bar[b.wh_out],'r')
    ax[0].set_xlabel('lat')
    ax[0].set_title('outbound')
    ax[1].plot(b.r[b.wh_out],b.bend[b.wh_out]*180/np.pi,'.')
    ax[1].plot(b.r[b.wh_out],bend_bar[b.wh_out],'r')
    ax[1].set_xlabel('R (RJ)')    
    plt.show()
    
    return bend_bar

def get_Bend_no_den(R,lat,Br,Btheta,Bphi,wh1,wh2):
    fig, ax = plt.subplots(2)
    bend = (Bphi*Br/(abs(Bphi)*np.abs(Br)))*myatan2(np.abs(Bphi),np.abs(Br))
    bend_bar = smooth(bend*180/np.pi,60)
    ax[0].plot(lat[wh1],bend[wh1]*180/np.pi,'.')
    ax[0].plot(lat[wh1],bend_bar[wh1],'r')
    ax[0].set_xlabel('lat')
    ax[0].set_ylabel('Bend angle (deg)')
    ax[0].set_title('inbound')
    ax[1].plot(R[wh1],bend[wh1]*180/np.pi,'.')
    ax[1].plot(R[wh1],bend_bar[wh1],'r')
    ax[1].set_xlabel('R (RJ)')
    ax[1].set_ylabel('Bend angle (deg)')
    ax[1].set_xlim([0,np.max(R[wh1])])
    """
    whh = nh_sig < 50*nh_den
    whp = np_sig < 50*np_den
    ax[2].plot(nh_r[whh],nh_den[whh],'.')
    ax[2].plot(np_r[whp],np_den[whp],'.')
    ax[2].set_xlim([0,np.max(R[wh1])])
    ax[2].get_shared_x_axes().join(ax[1], ax[2])
    ax[2].set_yscale('log')
    ax[2].set_xlabel('R (RJ)')
    ax[2].set_ylabel('Heavy density (cc)')
    plt.show()
    """

    fig, ax = plt.subplots(2)
    bend = Bphi*Br/(abs(Bphi)*np.abs(Br))*myatan2(np.abs(Bphi),np.abs(Br))
    bend_bar = smooth(bend*180/np.pi,60)
    ax[0].plot(lat[wh2],bend[wh2]*180/np.pi,'.')
    ax[0].plot(lat[wh2],bend_bar[wh2],'r')
    ax[0].set_xlabel('lat')
    ax[0].set_title('outbound')
    ax[1].plot(R[wh2],bend[wh2]*180/np.pi,'.')
    ax[1].plot(R[wh2],bend_bar[wh2],'r')
    ax[1].set_xlabel('R (RJ)')    
    plt.show()
    
    return bend,bend_bar


def get_corr_2(n,b,rmin=30):
    print('rmin...',rmin)
    #btot = np.sqrt(b.Br**2 + b.Btheta**2 + b.Bphi**2)
    Bdp = Jup_dipole(b.r)
    plt.figure()
    plt.plot(b.Btheta[b.wh_in]/b.btot[b.wh_in],b.bend[b.wh_in],'.')
    plt.plot(b.Btheta[b.wh_out]/b.btot[b.wh_out],b.bend[b.wh_out],'.')
    #plt.plot(b.Btheta[b.wh_in]/Bdp[b.wh_in],b.bend[b.wh_in],'.')
    #plt.plot(b.Btheta[b.wh_out]/Bdp[b.wh_out],b.bend[b.wh_out],'.')
    plt.xlabel('$B_\\theta/B$')
    plt.ylabel('Bend angle (deg)')
    plt.show()

    #plt.figure()
    #plt.plot(n.h/n.p,'.')
    #plt.yscale('log')
    brat = []
    nrat = []
    denrat = []
    trat = []
    latrat = []
    Rrat = []
    bendrat = []
    tarr = b.t[b.wh_in]
    latarr = b.lat[b.wh_in]
    Rarr = b.r[b.wh_in]
    barr = b.Btheta[b.wh_in]/b.btot[b.wh_in]
    #barr = b.Btheta[b.wh_in]/Bdp[b.wh_in]
    bendarr = b.bend[b.wh_in]
    
    for i in range(len(n.t)):
        idx = (np.abs(tarr-n.t[i])).argmin()
        if idx < len(tarr)-1:
            brat.append(barr[idx])
            trat.append(tarr[idx])
            denrat.append(n.h[i]/n.p[i])
            latrat.append(latarr[idx])
            Rrat.append(Rarr[idx])
            bendrat.append(bendarr[idx])
    print('Orbit time...',np.max(np.asarray(trat))-np.min(np.asarray(trat)))
    denrat = np.asarray(denrat)
    brat = np.asarray(brat)
    Rrat = np.asarray(Rrat)
    bendrat = np.asarray(bendrat)
    plt.show()
    plt.figure()
    #wh = np.logical_and(Rrat > 30, Rrat < 50)
    wh = np.where(Rrat > rmin)
    plt.plot(denrat[wh],brat[wh],'.')
    plt.xlim([1e-1,100])
    plt.xscale('log')
    plt.xlabel('$n_h/n_p$')
    plt.ylabel('$B_\\theta/|B|$')
    #plt.ylabel('Bend angle')
    whneg = np.logical_and(brat < -0.0, denrat > 1e-2)
    whneg2 = np.where(whneg == True)
    print(len(whneg2[0]))
    plt.title('$B_\\theta$ < 0 occurrence freq: '+str(100*len(whneg2[0])/len(brat))[:5]+'%')
    plt.show()
    
    """
    plt.figure()
    #wh = np.logical_and(brat < -0.05, denrat > 1e-2)
    plt.hist(denrat[whneg],bins=20)
    plt.xlabel('$n_h/n_p$')
    plt.ylabel('counts')
    plt.title('$B_\\theta$ < 0 occurrence freq: '+str(100*len(whneg2[0])/len(brat))[:5]+'%')
    #plt.xscale('log')
    #plt.plot(denrat,'.')
    plt.show()
    """
    
    oc_freq = 100*len(whneg2[0])/len(brat)

    plt.figure()
    plt.plot(Rrat,denrat,'.')
    plt.ylim([0,10])
    #plt.xlim()
    #plt.xscale('log')
    plt.xlabel('lat')
    plt.ylabel('$n_h/n_p$')
    #plt.xlabel('$B_\\theta/|B|$')
    plt.show()
        
    return oc_freq

def get_bend_events(n,b,dt,orbit):
    f = open('Bend_events'+str(orbit)+'.txt',"w")
    bend_bar = smooth(b.bend*180/np.pi,dt)
    bmean = smooth(b.btot,dt)
    wh = np.logical_and(np.logical_and(bend_bar > 0,b.bc_id == 1), np.logical_and(b.Btheta > 0.5*bmean, b.r > 40))
    #wh1 = np.where(np.logical_and(bend_bar > 0, b.Btheta > 0.5*bmean))
    #wh = np.logical_and(np.logical_and(wh, b.wh_out),b.r > 40)
#    wh1 = np.logical_and(wh, b.r > 40)
    wh1 = np.where(wh)
    wh1 = wh1[0]

    #wh = np.where(b.bend[0:])
    #start = b.t
    #wh = np.where(np.logical_and(bend_bar > 0.0, b.r > 40))
#    plt.figure()
    fig, ax = plt.subplots(2)
    ax[0].plot(b.t[wh1],bend_bar[wh1],'o')
    ax[0].set_ylabel('bend_bar')
    print(len(wh1))
    i = 0
    f.write('ORBIT, START_TIME, END_TIME, RADIAL DIST (RJ), LAT, DELTA_T (MIN)\n')
    while i < len(wh1)-1:    
    #for i in range(len(wh1)-1):
        flg = True
        j = i
        while np.logical_and(flg == True, j < len(wh1)-1):
            #deltat = abs(b.t[wh1[i]] - b.t[wh1[j]]).total_seconds()/60
            deltat1 = abs(b.t[wh1[j]] - b.t[wh1[j+1]]).total_seconds()
            deltat0 = abs(b.t[wh1[i]] - b.t[wh1[j]]).total_seconds()
            if (deltat1 > 60):
                if(deltat0 >=5*60):
                    f.write(str(orbit) + ', ' + str(b.t[wh1[i]]) + ', ' + str(b.t[wh1[j]]) + ', ' + str(b.r[wh1[i]]) +', ' + str(b.lat[wh1[i]]) + ', ' + str(abs(b.t[wh1[i]] - b.t[wh1[j]]).total_seconds()/60)+'\n')
#                    print(str(orbit) + ', ' + str(b.t[wh1[i]]) + ', ' + str(b.t[wh1[j]]) + ', ' + str(b.r[wh1[i]]) +', ' + str(b.lat[wh1[i]]) + ', ' + str(abs(b.t[wh1[i]] - b.t[wh1[j]]).total_seconds()/60), sep="")
                flg = False
            j += 1
        i = j
            
    ax[0].plot(b.t[wh],b.bend[wh]*180/np.pi,'.')
#    ax[1].plot(n.t,n.h,label='heavy')
#    ax[1].plot(n.t,n.p,label='protons')

#    ax[1].set_yscale('log')
#    ax[1].legend(loc="best")
    wh = np.where(b.bc_id == 1)
    ax[1].plot(b.t[wh],b.Br[wh],label='$B_r$')
    ax[1].plot(b.t[wh],b.Btheta[wh],label='$B_\\theta$')
    ax[1].plot(b.t[wh],b.Bphi[wh],label='$B_\phi$')
    ax[1].plot(b.t[wh],b.btot[wh],'k',label='|B|')
    ax[1].plot(b.t[wh],-b.btot[wh],'k',label='|B|')
    ax[1].legend(loc="best")
    ax[1].set_ylim([-50,50])
    ax[1].get_shared_x_axes().join(ax[0], ax[1])
    plt.show()
    f.close()
    #print(b.t[wh])
    

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

        
#    timeStart = '2017-05-15T22:55:48'
#    timeEnd = '2017-05-16T22:55:48'

    timeStart = orbitsData[orbit]
#    timeEnd = '2017-05-17T00:00:00'
    timeEnd = orbitsData[orbit+1]

    print(timeStart,timeEnd)
    
    Rj = 7.14e4
    
    b = MagData(timeStart,timeEnd,'/data/juno_spacecraft/data/fgm_ss',['fgm_jno','r60s'])    

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
          
    n = DenClass(timeStart,timeEnd)
    n.read_density()
    
    #j = JadClass(jad_tm,jad_arr,jad_mean)
    
    bx = b.data_df['BX'][timeStart:timeEnd]
    by = b.data_df['BY'][timeStart:timeEnd]
    bz = b.data_df['BZ'][timeStart:timeEnd]
    x = b.data_df['X'][timeStart:timeEnd]
    y = b.data_df['Y'][timeStart:timeEnd]
    z = b.data_df['Z'][timeStart:timeEnd]
    
    bc_df = get_mp_bc()
    bc_id = get_bc_mask(bx,bc_df)
    
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

    b = MagClass(Bx,By,Bz,Br,Btheta,Bphi,bc_id,wh1,wh2,R,lat,t)
    
    #get_B_den_profiles(t,r,Br,Btheta,Bphi,wh,np_den,nh_den,np_sig,nh_sig)
    
    #oc_freq = get_corr(t,R,lat,Br,Btheta,Bphi,wh1,wh2,bend,nh_den,np_den,nh_r,np_r)
    #bend, bend_bar = get_Bend_no_den(R,lat,Br,Btheta, Bphi,wh1,wh2)    

    return n,b,j

#----------------------------------------------------------------------------------------

#for i in range(6,32):
    
orbit = 12
n,b,j = plot_Juno_mag(orbit)
wh = (b.r > 10) & (b.bc_id == 1)
whjad = (j.R > 10) & (j.bc_id == 1)
#wh = b.r > 10
    #get_B_profiles_2(b,wh)
#get_B_dB_profiles_2(b,10,wh,whjad,j,n,orbit)
#plt.show()

get_dn_dB_profiles(b,10,wh,whjad,j,n,orbit)
#get_jade_variation()


#get_B_den_profiles_2(n,b,wh)

#bend_bar = get_Bend_2(n,b)
#bend_ave = bend_bar[np.logical_not(np.isnan(bend_bar))]
#bend_r = b.r[np.logical_not(np.isnan(bend_bar))]

#rmax = 50
#bend_plus = bend_ave[np.logical_and(bend_ave > 0, bend_r > rmax)]
#bend_r_plus = bend_r[np.logical_and(bend_ave > 0, bend_r > 80)]
#print('bend plus...',100*len(bend_plus)/len(bend_ave[bend_r > rmax]))
#oc_freq = get_corr_2(n,b,30)
#get_bend_events(n,b,10,orbit)
"""
plt.figure()
plt.hist(bend_bar,bins=40,alpha=1.0)
#    plt.hist([bend*180/np.pi,bend_bar],bins='auto',alpha=0.5)
plt.xlabel('Bend angle (deg)')
plt.ylabel('Count')
plt.title('orbit...'+str(orbit))
plt.yscale('log')
plt.show()
"""
   


"""
oc_arr = []
orbit_arr = []
bend_arr = []
bend_plus_arr = []
for i in range(5,27):
    print('orbit...',i)
    bend_ave,bend_r,n,b = plot_Juno_mag(i)
    bend_plus = bend_ave[bend_ave > 0]
    print('bend_ave > 0....',100*len(bend_plus)/len(bend_ave))
    oc_freq = get_corr_2(n,b,40)
    oc_arr.append(oc_freq)
    orbit_arr.append(i)
    bend_arr.append(bend_ave.mean())
    bend_plus_arr.append(100*len(bend_plus)/len(bend_ave))
    
plt.figure()
plt.plot(orbit_arr,oc_arr)
#plt.plot(orbit_arr,smooth(oc_arr,2))
plt.xlabel('orbit')
plt.ylabel('$B_\\theta < 0$ occurrence freq (%)')
plt.show()

plt.figure()
plt.plot(orbit_arr,bend_arr)
#plt.plot(orbit_arr,smooth(bend_arr,2))
plt.xlabel('orbit')
plt.ylabel('<bend angle>')
plt.show()

plt.figure()
plt.plot(orbit_arr,bend_plus_arr)
#plt.plot(orbit_arr,smooth(bend_arr,2))
plt.xlabel('orbit')
plt.ylabel('bend plus occur freq (%)')
plt.show()
"""


