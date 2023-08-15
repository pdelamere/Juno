
import re
import pickle
import struct
from datetime import datetime, timedelta
from os import fsdecode
import pathlib

import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import pandas as pd
import pycwt as wavelet
import scipy.integrate as integrate
import scipy.signal as signal
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spiceypy.spiceypy import xf2eul
from sklearn.linear_model import LinearRegression
import math as m

from juno_functions import _get_files, time_in_lat_window, get_sheath_intervals

myatan2 = np.vectorize(m.atan2)
Rj = 7.14e4

class PlotClass:
    def __init__(self, axes, xlabel=None, ylabel=None, title=None):
        self.axes = axes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def xaxis_datetime_tick_labels(self, x_ticks_labeled,):
        
        locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        self.axes.xaxis.set_major_locator(locator)
        self.axes.xaxis.set_major_formatter(formatter)
        if not x_ticks_labeled:
            self.axes.set_xticklabels([])

    def plot(self, x, y, magnitude=False, data_labels=None, xlabel=None, ylabel=None,
                         title=None, **kwargs):

        if (np.ndim(y) == 1) & (np.ndim(x) == 1):
            self.axes.plot(x, y, **kwargs)
        elif (np.ndim(y) != 1) & (np.ndim(x) != 1):
            self.axes.plot(np.transpose(x), np.transpose(y), **kwargs)
        else:
            if data_labels is None:
                self.axes.plot(x, np.transpose(y), **kwargs)
            else:
                for i in range(len(y)):
                    self.axes.plot(x, y[i], label=data_labels[i], **kwargs)
        
        if magnitude:
            mag = np.array([np.sqrt(np.sum(np.power(i, 2))) for i in np.transpose(y)])
            self.axes.plot(x, mag, label='Magnitude', color='black', **kwargs)
            self.axes.plot(x, -mag, color='black', **kwargs)

        if magnitude and data_labels != None:
            self.axes.legend(loc='upper left')
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_title(self.title)

    def colormesh(self, x, y, data, color_bar=True, xlabel=None, ylabel=None, title=None, **kwargs):

        cwt = self.axes.pcolormesh(x, y, data, shading='auto', **kwargs)
        self.axes.set_title(title)
        self.axes.set_ylim(y[0], y[-1])
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlabel(xlabel)
        self.axes.set_xlim(x[0], x[-1])
        if color_bar:
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", '2%', pad='2%')
            cbr = plt.colorbar(cwt, cax=cax)
            cbr.set_label('Magnitude', rotation=270, labelpad=5, size=8)

class PosData():
    def __init__(self, datetime_series):
        self.datetime_series = datetime_series
        
    def JSS_Pos_data(self):
        pass
    
    def sys_3(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
            
        et_array = [spice.utc2et(i) for i in self.datetime_series.strftime('%Y-%m-%dT%H:%M:%S')]
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
        z_equator = positions.T[2]/7.14e4 - R*np.sin(CentEq2)


class bc_ids:
    
    def get_mp_bc(self):
        bc_df = pd.read_csv('./wholecross5.csv')
        #bc_df = pd.read_csv('./crossevent7.csv')
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
        #positions, lt = spice.spkpos('JUNO', et_array,
        #                             'IAU_JUPITER', 'NONE', 'JUPITER')
        positions, lt = spice.spkpos('JUNO', et_array,
                                     'JUNO_MAG_VIP4', 'NONE', 'JUPITER')
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
        Req = np.sqrt(x**2 + y**2)/7.14e4
        c = 1.62
        d = 7.76*deg2rad
        e = 249*deg2rad
        CentEq2 = (a*np.tanh(b*R - c) + d)*np.sin(lon*deg2rad - e)
        #self.z_cent = positions.T[2]/7.14e4 - R*np.sin(CentEq2)
        self.z_cent = z/7.14e4
        self.R = R
        self.Req = Req

        """
        for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        
        index_array = self.t #self.jad_tm
        et_array = [spice.utc2et(i) for i in index_array.strftime('%Y-%m-%dT%H:%M:%S')]
        positions, lt = spice.spkpos('JUNO', et_array,
                                     'JUNO_JSS', 'NONE', 'JUPITER')
        x = np.array(positions.T[0])
        y = np.array(positions.T[1])
        z = np.array(positions.T[2])
        """
        
        return x,y,z

    def Jup_dipole(self,r):
        B0 = 417e-6/1e-9   #nTesla
        Bdp = B0/r**3 #r in units of planetary radius
        return Bdp

    def smooth(self,y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

        
class WavData():
    """Collects and stores all mag data between two datetimes.

    Attributes
    ----------
    start_time : string
        Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
    end_time : string
        End datetime in ISO format. e.g. "2016-01-01T00:00:00"
    data_df : string
        Pandas dataframe containing magnitometer data indexed by a DatetimeIndex.
    data_files : list
        List of filepaths to data files containing data between the two datetimes.
        This is gotten using an internal function

    """

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/wav',
                 instrument=['WAV_', '_E_V02']):
        """Find and store all data between two datetimes.

        Parameters
        ----------
        start_time : string
            Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
           End datetime in ISO format. e.g. "2016-01-01T00:00:00"
        data_folder : str, optional
            Path to folder containing csv data files. The default is '/data/juno_spacecraft/data'.
        instrument : list of strings, optional
            List of strings that will be in filenames to aid file search.
                The default is ['fgm_jno', 'r1s'].

        Returns
        -------
        None.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.data_files = self._get_files('CSV', data_folder, *instrument)
        print('wav data files...',self.data_files)
        self.data_df = pd.DataFrame()
        self.freq = 0.0
        self.t = 0.0
        self._get_data()

    def _get_files(self,file_type, data_folder, *args):
        import os
        """Find all files between two dates.

        Parameters
        ----------
        start_time : string
        start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
        end datetime in ISO format. e.g. "2016-01-01T00:00:00"
        file_type : string
        The type of file the magnetometer data is stored in. e.g. ".csv"
        data_folder : string
        folder which all data is stored in.
        *args : string
        strings in filenames that wil narow down searching.
        
        Returns
        -------
        file_paths : list
        List of paths to found files.
        
        """
    
        if file_type.startswith('.'):
            pass
        else:
            file_type = '.' + file_type
            datetime_array = pd.date_range(self.start_time, self.end_time, freq='D').date
            #print(datetime_array)
            file_paths = []
            file_dates = []
            date_re = re.compile(r'\d{7}')
            instrument_re = re.compile('|'.join(args))
            for parent, child, files in os.walk(data_folder):
                for file_name in files:
                    if file_name.endswith(file_type):
                        
                        file_path = os.path.join(parent, file_name)
                        file_date = datetime.strptime(
                            date_re.search(file_name).group(), '%Y%j')
                        instrument_match = instrument_re.findall(file_name)
                  
                        if file_date.date() in datetime_array and sorted(args) == sorted(instrument_match):

                            file_paths = np.append(file_paths, file_path)
                            file_dates = np.append(file_dates, file_date)
                            
                            sorting_array = sorted(zip(file_dates, file_paths))
                            file_dates, file_paths = zip(*sorting_array)
            del(datetime_array, file_dates)
                        
            return file_paths
        
        
    def _get_data(self):
        for wav_csv in self.data_files:
            print('opening files....',wav_csv)
            csv_df = pd.read_csv(wav_csv,skiprows=2)
            csv_df.drop(csv_df.columns[0],axis=1,inplace=True)
            csv_df.drop(csv_df.columns[1:27],axis=1,inplace=True)
            freq = csv_df.iloc[0,1:]
            #print(freq)
            csv_df.drop([0,1],axis=0,inplace=True)
            csv_df.rename(columns={csv_df.columns[0]: "DATETIME"},inplace=True)
            csv_df['DATETIME'] = pd.to_datetime(csv_df['DATETIME'],format='%Y-%jT%H:%M:%S.%f')
            
            #csv_df['DATETIME'] = csv_df['DATETIME'].astype('datetime64[ns]')

            csv_df = csv_df.set_index('DATETIME')
            
            
            csv_df.index = csv_df.index.astype('datetime64[ns]').floor('S')
            #print(csv_df.index)
            self.data_df = self.data_df.append(csv_df)
            #print(self.data_df)
            self.data_df = self.data_df.sort_index()
            #print(self.data_df.index, csv_df.index)
            self.data_df = self.data_df[self.start_time: self.end_time].sort_index()
            #print(self.data_df.info())
        self.data_df = self.data_df.iloc[::600,:]
        self.freq = freq
        self.t = self.data_df.index
        del csv_df

    def plot_wav_data(self,thres):
        from matplotlib import ticker
        arr = self.data_df.to_numpy()
        arr = arr.transpose()
        #plt.contourf(self.t,self.freq,arr.transpose(),levels=50,locator=ticker.LogLocator(),vmin=1e-14, vmax=1e-9)
        vmin = 1e-14
        vmax = 1e-10
        lev = np.linspace(np.log(vmin),np.log(vmax),10)
        #plt.pcolor(self.t,self.freq[1:35],arr[1:35,:],norm=LogNorm())
        plt.pcolormesh(self.t,self.freq[1:40],arr[1:40,:],norm=LogNorm(vmin=5e-15, vmax=1e-10))
        #plt.contourf(self.t,self.freq,arr.transpose(),levels=np.exp(lev),norm = LogNorm())
        #plt.imshow(arr.transpose(), norm=LogNorm(),aspect='auto',origin='lower') 
        plt.yscale('log')
        plt.ylabel('freq (Hertz)')
        plt.xlabel('time')
        #plt.ylim([1e2,2e4])
        plt.colorbar()
        plt.show()

class JAD_MOM_Data(bc_ids):
    """Collects and stores all mag data between two datetimes.

    Attributes
    ----------
    start_time : string
        Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
    end_time : string
        End datetime in ISO format. e.g. "2016-01-01T00:00:00"
    data_df : string
        Pandas dataframe containing magnitometer data indexed by a DatetimeIndex.
    data_files : list
        List of filepaths to data files containing data between the two datetimes.
        This is gotten using an internal function

    """

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/jad_moments/AGU2020_moments',
                 instrument=['PROTONS', 'V03']):
        """Find and store all data between two datetimes.

        Parameters
        ----------
        start_time : string
            Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
           End datetime in ISO format. e.g. "2016-01-01T00:00:00"
        data_folder : str, optional
            Path to folder containing csv data files. The default is '/data/juno_spacecraft/data'.
        instrument : list of strings, optional
            List of strings that will be in filenames to aid file search.
                The default is ['fgm_jno', 'r1s'].

        Returns
        -------
        None.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.data_files = self._get_files('CSV', data_folder, *instrument)
        self.data_df = pd.DataFrame()
        self.t = 0.0
        self.n = 0.0
        self.n_sig = 0.0
        self.vr = 0.0
        self.vr_sig = 0.0
        self.vtheta = 0.0
        self.vtheta_sig = 0.0
        self.vphi = 0.0
        self.vphi_sig = 0.0
        self.T = 0.0
        self.T_sig = 0.0
        self._get_data()
        self.bc_id = 0.0
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()
        x,y,z = self.sys_3_data()
        self.x = x
        self.y = y
        self.z = z
        #self.plot_jad_data()

    def _get_files(self,file_type, data_folder, *args):
        import os
        """Find all files between two dates.

        Parameters
        ----------
        start_time : string
        start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
        end datetime in ISO format. e.g. "2016-01-01T00:00:00"
        file_type : string
        The type of file the magnetometer data is stored in. e.g. ".csv"
        data_folder : string
        folder which all data is stored in.
        *args : string
        strings in filenames that wil narow down searching.
        
        Returns
        -------
        file_paths : list
        List of paths to found files.
        
        """
    
        if file_type.startswith('.'):
            pass
        else:
            file_type = '.' + file_type
            datetime_array = pd.date_range(self.start_time, self.end_time, freq='D').date
            #print(datetime_array)
            file_paths = []
            file_dates = []
            date_re = re.compile(r'\d{7}')
            instrument_re = re.compile('|'.join(args))
            for parent, child, files in os.walk(data_folder):
                for file_name in files:
                    if file_name.endswith(file_type):
                        
                        file_path = os.path.join(parent, file_name)
                        file_date = datetime.strptime(
                            date_re.search(file_name).group(), '%Y%j')
                        instrument_match = instrument_re.findall(file_name)
                  
                        if file_date.date() in datetime_array and sorted(args) == sorted(instrument_match):

                            file_paths = np.append(file_paths, file_path)
                            file_dates = np.append(file_dates, file_date)
                            
                            sorting_array = sorted(zip(file_dates, file_paths))
                            file_dates, file_paths = zip(*sorting_array)
            del(datetime_array, file_dates)
                        
            return file_paths
        
        
    def _get_data(self):
        for jad_csv in self.data_files:
            print('opening files....',jad_csv)
            csv_df = pd.read_csv(jad_csv)
            csv_df.rename(columns={csv_df.columns[0]: "DATETIME"},inplace=True)
            csv_df['DATETIME'] = pd.to_datetime(csv_df['DATETIME'],format='%Y-%jT%H:%M:%S.%f')
            
            csv_df = csv_df.set_index('DATETIME')
            
            csv_df.index = csv_df.index.astype('datetime64[ns]').floor('S')
            self.data_df = self.data_df.append(csv_df)
            self.data_df = self.data_df.sort_index()
            self.data_df = self.data_df[self.start_time: self.end_time].sort_index()
        self.data_df.rename(columns={"N_CC": "n", "N_SIGMA_CC": "n_sig", "V_JSSRTP_KMPS[0]": "vr",
                                     "V_JSSRTP_SIGMA_KMPS[0]": "vr_sig", "V_JSSRTP_KMPS[1]": "vtheta", 
                                     "V_JSSRTP_SIGMA_KMPS[1]": "vtheta_sig", "V_JSSRTP_KMPS[2]": "vphi", 
                                     "V_JSSRTP_SIGMA_KMPS[2]": "vphi_sig", "TEMP_EV": "Temp",
                                     "TEMP_SIGMA_EV": "Temp_sig"}, inplace=True)
        self.t = self.data_df.index
        """                    
        self.n = self.data_df.N_CC
        self.nave = self.data_df.N_CC.rolling(10).mean()
        self.n_sigma = self.data_df.N_SIGMA_CC
        self.vr = self.data_df['V_JSSRTP_KMPS[0]']
        self.vr_sigma = self.data_df['V_JSSRTP_SIGMA_KMPS[0]']
        self.vphi = self.data_df['V_JSSRTP_KMPS[1]']
        self.vphi_sigma = self.data_df['V_JSSRTP_SIGMA_KMPS[1]']
        self.vtheta = self.data_df['V_JSSRTP_KMPS[2]']
        self.vtheta_sigma = self.data_df['V_JSSRTP_SIGMA_KMPS[2]']
        self.T = self.data_df['TEMP_EV']
        self.T_sigma = self.data_df['TEMP_SIGMA_EV']
        """
        del csv_df

    def plot_jad_data(self,sig_max,win,species):
        #from matplotlib import ticker
        wh = (self.data_df.n_sig < sig_max) & (self.data_df.n > 0)
        fig, ax = plt.subplots(5,1,sharex=True)
        fig.set_size_inches((12,8))
        ax[0].set_title(species)
        ax[0].plot(self.data_df.n[wh])
        ax[0].plot(self.data_df.n[wh].rolling(win).mean())
        ax[0].set_yscale('log')
        #ax[0].plot(self.data_df.n[wh].rolling(win).mean())
        wh = (self.data_df.vr_sig < sig_max)
        ax[1].plot(self.data_df.vr[wh],label='vr')
        ax[1].plot(self.data_df.vr[wh].rolling(win).mean())
        ax[1].set_ylim([-500,500])
        ax[1].set_ylabel('vr')
        wh = (self.data_df.vtheta_sig < sig_max)
        ax[2].plot(self.data_df.vtheta[wh],label='vr')
        ax[2].plot(self.data_df.vtheta[wh].rolling(win).mean())
        ax[2].set_ylim([-500,500])
        ax[2].set_ylabel('vtheta')
        wh = (self.data_df.vphi_sig < sig_max)
        ax[3].plot(self.data_df.vphi[wh],label='vr')
        ax[3].plot(self.data_df.vphi[wh].rolling(win).mean())
        ax[3].set_ylim([-500,500])
        ax[3].set_ylabel('vphi')
        #ax[1].plot(self.data_df.vtheta,label='vtheta')
        #ax[1].plot(self.data_df.vphi,label='vphi')
        #ax[1].legend(loc="best")
        wh = (self.data_df.Temp > 0) & (self.data_df.Temp_sig < sig_max)
        ax[4].plot(self.data_df.Temp[wh])
        ax[4].plot(self.data_df.Temp[wh].rolling(win).mean())
        ax[4].set_yscale('log')
        ax[4].set_ylabel('Temp') 
        #plt.show()
        #plt.plot(self.data_df.n[wh].rolling(win).mean())


class JEDI_MOM_h5(bc_ids):

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/jedi_moments',
                 instrument=['OpS']):
        
        self.start_time = start_time
        self.end_time = end_time
        print(self.start_time,self.end_time)
        self.data_files = self._get_files('h5', data_folder, *instrument)
        print(self.data_files)
        self.R = 0.0
        self.Req = 0.0
        self.z_cent = 0.0
        
        self.Density = np.empty(0)
        self.P = np.empty(0)
        self.PA = np.empty(0)
        self.Year = np.empty(0)
        self.Month = np.empty(0)
        self.DOM = np.empty(0)
        self.Hour = np.empty(0)
        self.Min = np.empty(0)
        self.Sec = np.empty(0)
        self.R = np.empty(0)
        self.t = 0.0
        self.data_df = pd.DataFrame()
        self._get_data()
        self.bc_id = 0.0
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()
        x,y,z = self.sys_3_data()
        self.x = x
        self.y = y
        self.z = z
        
    def _get_files(self,file_type, data_folder, *args):
        import os
        """Find all files between two dates.
        
        Parameters
        ----------
        start_time : string
        start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
        end datetime in ISO format. e.g. "2016-01-01T00:00:00"
        file_type : string
        The type of file the magnetometer data is stored in. e.g. ".csv"
        data_folder : string
        folder which all data is stored in.
        *args : string
        strings in filenames that wil narow down searching.
        
        Returns
        -------
        file_paths : list
        List of paths to found files.
        
        """
    
        if file_type.startswith('.'):
            pass
        else:
            file_type = '.' + file_type
            datetime_array = pd.date_range(self.start_time, self.end_time, freq='D').date
            file_paths = []
            file_dates = []
            date_re = re.compile(r'\d{4}\-\d{2}\-\d{2}')
            instrument_re = re.compile('|'.join(args))
                        
            for parent, child, files in os.walk(data_folder):
                for file_name in files:
                    
                    if file_name.endswith(file_type):
                        file_path = os.path.join(parent, file_name)
                        
                        file_date = datetime.strptime(
                            date_re.search(file_name).group(), "%Y-%m-%d")
                        instrument_match = instrument_re.findall(file_name)
                        if sorted(args) == sorted(instrument_match):
                           
                            file_paths = np.append(file_paths, file_path)
                            file_dates = np.append(file_dates, file_date)
                            
                            sorting_array = sorted(zip(file_dates, file_paths))
                            file_dates, file_paths = zip(*sorting_array)
            del(datetime_array, file_dates)
            #print("file_paths...",file_paths)          
            return file_paths
        
    def _get_data(self):
        import h5py

        for jedi_ in self.data_files:
            #tmp_df = pd.DataFrame(columns = ["Density", "P", "R"])

            t_df = pd.DataFrame()
            f = h5py.File(jedi_, 'r')
            #print('columns...',f.keys())
            self.Density = np.append(self.Density, f['Density'][0:])
            self.P = np.append(self.P, f['P'][0:])
            self.Year = np.append(self.Year, f['Year'][0:])
            self.Month = np.append(self.Month, f['Month'][0:])
            self.DOM = np.append(self.DOM, f['DOM'][0:])
            self.Hour = np.append(self.Hour, f['Hour'][0:])
            self.Min = np.append(self.Min, f['Min'][0:])
            self.Sec = np.append(self.Sec, f['Sec'][0:])
            self.R = np.append(self.R, f['JSO-R'][0:])
        self.t = []            
        for i in range(len(self.Year)):
            self.t.append(datetime(year = int(self.Year[i]), month = int(self.Month[i]), day = int(self.DOM[i]), hour = int(self.Hour[i]),
                                   minute = int(self.Min[i]), second = int(self.Sec[i])))


        #self.t = np.asarray(self.t, dtype='datetime64')
        #print('JEDI time...',self.t)
        self.data_df.index = self.t
        self.data_df.index = self.data_df.index.astype('datetime64[ns]').floor('S')
        self.t = self.data_df.index
        #print('JEDI time...',self.t)
        self.data_df['Density'] = self.Density
        self.data_df['P'] = self.P
        self.data_df['R'] = self.R
        #print('data_df...',self.data_df)             
    
    
class JEDI_MOM_Data(bc_ids):
    """Collects and stores all mag data between two datetimes.

    Attributes
    ----------
    start_time : string
        Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
    end_time : string
        End datetime in ISO format. e.g. "2016-01-01T00:00:00"
    data_df : string
        Pandas dataframe containing magnitometer data indexed by a DatetimeIndex.
    data_files : list
        List of filepaths to data files containing data between the two datetimes.
        This is gotten using an internal function

    """

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/jedi_moments',
                 instrument=['heavy', 'p']):
        """Find and store all data between two datetimes.

        Parameters
        ----------
        start_time : string
            Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
           End datetime in ISO format. e.g. "2016-01-01T00:00:00"
        data_folder : str, optional
            Path to folder containing csv data files. The default is '/data/juno_spacecraft/data'.
        instrument : list of strings, optional
            List of strings that will be in filenames to aid file search.
                The default is ['fgm_jno', 'r1s'].

        Returns
        -------
        None.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.data_files = self._get_files('d2s', data_folder, *instrument)
        self.data_df = pd.DataFrame()
        self.t = 0.0
        self.n_elec = 0.0
        self.n_hp = 0.0
        self.n_heavy = 0.0
        self.p_elec = 0.0
        self.p_hp = 0.0
        self.p_heavy = 0.0
        self._get_data()
        self.bc_id = 0.0
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()
        x,y,z = self.sys_3_data()
        self.x = x
        self.y = y
        self.z = z
        #self.plot_jad_data()

    def _get_files(self,file_type, data_folder, *args):
        import os
        """Find all files between two dates.

        Parameters
        ----------
        start_time : string
        start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
        end datetime in ISO format. e.g. "2016-01-01T00:00:00"
        file_type : string
        The type of file the magnetometer data is stored in. e.g. ".csv"
        data_folder : string
        folder which all data is stored in.
        *args : string
        strings in filenames that wil narow down searching.
        
        Returns
        -------
        file_paths : list
        List of paths to found files.
        
        """
    
        if file_type.startswith('.'):
            pass
        else:
            file_type = '.' + file_type
            datetime_array = pd.date_range(self.start_time, self.end_time, freq='D').date
            #print(datetime_array)
            file_paths = []
            file_dates = []
            date_re = re.compile(r'\w{10}')

            instrument_re = re.compile('|'.join(args))
            for parent, child, files in os.walk(data_folder):
                for file_name in files:
                    if file_name.endswith(file_type):
                        
                        file_path = os.path.join(parent, file_name)
                        
                        file_date = datetime.strptime(
                            date_re.search(file_name).group(), "%Y_%m_%d")
                        instrument_match = instrument_re.findall(file_name)
                  
                        if file_date.date() in datetime_array and sorted(args) == sorted(instrument_match):
                            print("sorted args...",sorted(args),sorted(instrument_match))
                            file_paths = np.append(file_paths, file_path)
                            file_dates = np.append(file_dates, file_date)
                            
                            sorting_array = sorted(zip(file_dates, file_paths))
                            file_dates, file_paths = zip(*sorting_array)
            del(datetime_array, file_dates)
                        
            return file_paths
        
        
    def _get_data(self):
        for jedi_csv in self.data_files:
            print('opening files....',jedi_csv)
            #csv_df = pd.read_csv(jad_csv)
            csv_df = pd.read_fwf(jedi_csv, skiprows=9)
            csv_df.rename(columns={csv_df.columns[0]: "DATETIME"},inplace=True)
            csv_df.rename(columns={csv_df.columns[1]: "DATA"},inplace=True)
            csv_df['DATETIME'] = pd.to_datetime(csv_df['DATETIME'],format=':01:%Y-%m-%dT%H:%M:%S.%f')
            
            csv_df = csv_df.set_index('DATETIME')
            csv_df.index = csv_df.index.astype('datetime64[ns]').floor('S')
            self.data_df = self.data_df.append(csv_df)
            self.data_df = self.data_df.sort_index()
            self.data_df = self.data_df[self.start_time: self.end_time].sort_index()
        #self.data_df.rename(columns={"N_CC": "n", "N_SIGMA_CC": "n_sig", "V_JSSRTP_KMPS[0]": "vr",
        #                             "V_JSSRTP_SIGMA_KMPS[0]": "vr_sig", "V_JSSRTP_KMPS[1]": "vtheta", 
        #                             "V_JSSRTP_SIGMA_KMPS[1]": "vtheta_sig", "V_JSSRTP_KMPS[2]": "vphi", 
        #                             "V_JSSRTP_SIGMA_KMPS[2]": "vphi_sig", "TEMP_EV": "Temp",
        #                             "TEMP_SIGMA_EV": "Temp_sig"}, inplace=True)
        self.t = self.data_df.index

        del csv_df

        
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
        #b = MagData(self.timeStart,self.timeEnd,'/data/juno_spacecraft/data/pickled_mag_pos',['jno_mag_pos','v01'])    
        
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
        
        
class MagData:
    """Collects and stores all mag data between two datetimes.

    Attributes
    ----------
    start_time : string
        Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
    end_time : string
        End datetime in ISO format. e.g. "2016-01-01T00:00:00"
    data_df : string
        Pandas dataframe containing magnitometer data indexed by a DatetimeIndex.
    data_files : list
        List of filepaths to data files containing data between the two datetimes.
        This is gotten using an internal function

    """

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/fgm_ss',
                 instrument=['fgm_jno','r60s']):
        """Find and store all data between two datetimes.

        Parameters
        ----------
        start_time : string
            Start datetime in ISO format. e.g. "2016-01-01T00:00:00"
        end_time : string
           End datetime in ISO format. e.g. "2016-01-01T00:00:00"
        data_folder : str, optional
            Path to folder containing csv data files. The default is '/data/juno_spacecraft/data'.
        instrument : list of strings, optional
            List of strings that will be in filenames to aid file search.
                The default is ['fgm_jno', 'r1s'].

        Returns
        -------
        None.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.data_files = _get_files(start_time, end_time, '.csv', data_folder, *instrument)
        #self.data_files = _get_files(start_time, end_time, '.pkl', data_folder, *instrument)
        self.data_df = pd.DataFrame()
        self._get_data()
    """
    def _get_data(self):
        for mag_file in self.data_files:
            if mag_file.endswith('.csv'):
                csv_df = pd.read_csv(mag_file)
                csv_df = csv_df.drop(['DECIMAL DAY', 'INSTRUMENT RANGE'], axis=1)
                csv_df.columns = ['DATETIME', 'BX', 'BY', 'BZ', 'X', 'Y', 'Z']
                csv_df = csv_df.set_index('DATETIME')
                csv_df.index = csv_df.index.astype('datetime64[ns]').floor('S')
                self.data_df = self.data_df.append(csv_df)
                self.data_df = self.data_df.sort_index()
                self.data_df = self.data_df[self.start_time: self.end_time].sort_index()
                del csv_df
            elif mag_file.endswith('.pkl'):
                with open(mag_file, 'rb') as pikl:
                    self.data_df = pd.concat([self.data_df, pickle.load(pikl)],
                                             axis=0) 
                pikl.close()
    """
    
    def _get_data(self):
        for mag_csv in self.data_files:
            csv_df = pd.read_csv(mag_csv)
            #csv_df = csv_df.drop(['DECIMAL DAY', 'INSTRUMENT RANGE', 'X', 'Y', 'Z'], axis=1)
            csv_df = csv_df.drop(['DECIMAL DAY', 'INSTRUMENT RANGE'], axis = 1)
            csv_df.columns = ['DATETIME', 'BX', 'BY', 'BZ', 'X', 'Y', 'Z']
            csv_df = csv_df.set_index('DATETIME')
            csv_df.index = csv_df.index.astype('datetime64[ns]').floor('S')
            self.data_df = self.data_df.append(csv_df)
            self.data_df = self.data_df.sort_index()
            self.data_df = self.data_df[self.start_time: self.end_time].sort_index()
        del csv_df
    
            
    def plot(self, axes, start, end, data_labels, plot_magnitude=False, plot_title=None,
             xlabel=None, ylabel=None, time_per_major='12H', time_per_minor='1H',
             tick_label_format='%m-%d %H', x_ticks_labeled=True, **kwargs):
        """Plot data from the class dataframe.

        Parameters
        ----------
        axes : matplotlib.Axes
            matplotlib Axes object to plot to.
        start : string
            datetime to begin the plot at in ISO format.
        end : string
            datetime to end the plot at in ISO format.
        data_labels : list
            List of column names in the data_df variable to plot.
        plot_magnitude : bool, optional
            Plot the calculated magnitude of the given column names. The default is False.
        plot_title : string, optional
            Title to put on the plot, if no title leave as False. The default is False.
        xlabel : bool, optional
            Add x label, label is "Time". The default is True.
        ylabel : bool, optional
            Add y label, label is "Frequency (Hz)". The default is True.
        time_per_major : str, optional
            How many hours each major tick will appear, should be number followed by unit. e.g.'1H'
                Seconds: "s" Minutes: "min" Hour: "h" Day: "d" Month: "m"
            The default is '12H'.
        time_per_minor : string, optional
            Hours each minor tick will appear. The default is '1H'.
            Formatting similar to above.
        x_ticks_labeled : bool, optional
            Show labels on the x ticks. The default is True.
        **kwargs : TYPE
            keywords arguments to pass to matplotlib plot.

        Returns
        -------
        None.

        """
        plot_data = self.data_df[data_labels][start: end]
        x = plot_data.index
        y = plot_data.to_numpy().T
        
        
        plot_class = PlotClass(axes)
        plot_class.plot(x, y, magnitude=plot_magnitude, data_labels=data_labels,
                        title=plot_title, xlabel=xlabel, ylabel=ylabel, **kwargs)
        plot_class.xaxis_datetime_tick_labels(x_ticks_labeled)
        axes.set_xlim(x[0], x[-1])

    def downsample_data(self, downsampled_rate=60):
        """Downsamples data to larger time steps between samples.

        Parameters
        ----------
        downsampled_rate : int, optional
            Desired sample rate of data in seconds. The default is 60 seconds.

        Returns
        -------
        None.

        """
        mag_avg_df = pd.DataFrame()
        # This loop will average out the data to be at a lower sample rate
        num_steps = round((self.data_df.index[-1] - self.data_df.index[0])
                          / pd.Timedelta(seconds=downsampled_rate))
        end_datetime = self.data_df.index[0].replace(microsecond=0)
        for i in range(0, num_steps):
            start_datetime = end_datetime
            end_datetime = start_datetime + \
                pd.Timedelta(seconds=downsampled_rate)
            avg_datetime_index = pd.DatetimeIndex([
                (start_datetime + pd.Timedelta(seconds=int(downsampled_rate/2))).isoformat()
            ])
            temp_avg_df = pd.DataFrame(
                self.data_df[start_datetime.isoformat(): end_datetime.isoformat()].mean()
            ).transpose().set_index(avg_datetime_index)
            mag_avg_df = mag_avg_df.append(temp_avg_df).dropna()

        self.data_df = mag_avg_df.sort_index()
        self.data_df.index.name = 'DATETIME'
        self.data_df = self.data_df.dropna()
        del(avg_datetime_index, mag_avg_df, temp_avg_df)

    def mean_field_align(self, window_size=24):
        """Rotate magnetometer data into a mean-field-aligned coordinate system.
            Using the methods described by Khurana & Kivelson[1989]

        Parameters
        ----------
        window_size : int, optional
            The size of the window in minutes that is moved over data to average over.
                This should be EVEN to ensure times of MFA and regular data line up.
                The default is 24.

        Returns
        -------
        None.

        """
        # A windows of size 'window_size' in minutes is then moved along the data
        # An average inside of the window is found for each entry
        mean_mag_data = pd.DataFrame({'MEAN_BX': [], 'MEAN_BY': [], 'MEAN_BZ': []})
        finish_datetime = (self.data_df.index[-1]
        - timedelta(minutes=np.floor(window_size/2)))
        for datetime_index in self.data_df.index:
            start_datetime = datetime_index
            end_datetime = start_datetime + timedelta(minutes=window_size)
            mean_datetime_index = pd.DatetimeIndex([
                (start_datetime + timedelta(minutes=round(window_size/2))).isoformat()
            ])
            temp_mean = self.data_df[start_datetime.isoformat(): end_datetime.isoformat()].mean()
            mean_mag_data = mean_mag_data.append(pd.DataFrame(
                {'MEAN_BX': temp_mean.BX,
                 'MEAN_BY': temp_mean.BY,
                 'MEAN_BZ': temp_mean.BZ}, index=mean_datetime_index))
            
            if mean_datetime_index == finish_datetime:
                break
        # mean_mag_data and data_df are cut to align the time series of each
        # mag_data loses half of the time_window in the front
        # mean_mag_data loses half of the time window in the end
        # The two dataframes are then concatenated into one for simplicity
        # self.data_df = self.data_df.drop(self.data_df[: (self.data_df.index[0] +
        #     timedelta(minutes=round(window_size / 2) - 1)
        # ).isoformat()].index)
        self.data_df = self.data_df[mean_mag_data.index[0].isoformat(): 
                                    mean_mag_data.index[-1].isoformat()]
        self.data_df = pd.concat([self.data_df, mean_mag_data], axis=1)
        del mean_mag_data

        # The perturbation components of the mean field are found.
        # The method used is described in Khurana & Kivelson 1989
        axes_df = pd.DataFrame({'B_PAR': [], 'B_PERP1': [], 'B_PERP2': []})
        mean_vecs = self.data_df[['MEAN_BX', 'MEAN_BY', 'MEAN_BZ']]
        mean_magnitude = np.sqrt((mean_vecs**2).sum(axis=1))
        raw_data = self.data_df[['BX', 'BY', 'BZ']]
        raw_magnitude = np.sqrt((raw_data**2).sum(axis=1))
        for i in range(len(mean_vecs)):
            z_hat = mean_vecs.iloc[i] / mean_magnitude[i]
            unit_vec = raw_data.iloc[i] / raw_magnitude[i]
            cross = np.cross(z_hat, unit_vec)
            y_hat = cross / (np.sqrt(np.sum(np.power(cross, 2))))
            x_hat = np.cross(y_hat, z_hat)
            temp_bx = np.dot((raw_data.iloc[i].to_numpy() - mean_vecs.iloc[i].to_numpy()), x_hat)
            temp_by = np.dot((raw_data.iloc[i].to_numpy() - mean_vecs.iloc[i].to_numpy()), y_hat)
            temp_bz = np.dot((raw_data.iloc[i].to_numpy() - mean_vecs.iloc[i].to_numpy()), z_hat)

            new_df_index = pd.DatetimeIndex([self.data_df.index[i].isoformat()])
            axes_df = axes_df.append(
                pd.DataFrame({'B_PAR': temp_bz, 'B_PERP1': temp_by, 'B_PERP2': temp_bx},
                             index=new_df_index)
            )
        self.data_df = pd.concat([self.data_df, axes_df], axis=1)
        self.data_df.index.name = 'DATETIME'
        self.data_df = self.data_df.dropna()
        del axes_df

class CWTData:
    def __init__(self, datetime_series, signal, dt, min_freq=None, max_freq=None,
                 wave_resolution=6, mother=wavelet.Morlet):
        """Class for calculating, manipulating, and plotting a continuous wavelet
        analysis of a signal.

        Parameters
        ----------
        datetime_series : array of datetime64[ns] data
            Array of datetime variables accompanying the signal data
        signal : array
            Signal to be analyzed
        dt : int
            Time in seconds between data samples in the signal
        min_freq : float
            Lowest frequency in the frequency range to calculate the cwt in.
            Leave as None to calculate over whole range.
        max_freq : float
            Highest frequency in the frequency range to calculate the cwt in.
            Leave as None to calculate over whole range.
        wave_resolution : int, optional
            Resolution to be used in the wavelet packet, by default 6
        mother : pycwt mother wavelet class, optional
            Wavelet class from the pycwt module, by default pycwt.Morlet

        """        
        self.time_series = datetime_series
        self.data = signal
        self.wave_resolution = wave_resolution
        self.dt = dt
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mother = mother
        self.peaks_found = False
        self._get_cwt_matrix()

    def _get_cwt_matrix(self):

        N = len(self.data)
        Fs = 1 / self.dt
        f = np.arange(0, N / 2 + 1) * Fs / N
        if self.min_freq is None:
            min_index = 1
        else:
            min_index = min(range(len(f)), key=lambda i: abs(f[i] - self.min_freq)) - 1
        if self.max_freq is None:
            max_index = len(self.data)
        else:
            max_index = min(range(len(f)), key=lambda i: abs(f[i] - self.max_freq))
        f = f[min_index: max_index]
        wave, scales, self.freqs, self.coi, fft, fftfreqs = wavelet.cwt(
            self.data, self.dt, self.mother(self.wave_resolution), freqs=f
        )
        self.power = np.abs(wave)**2
        self.coi = self.coi**-1

    def remove_coi(self):
        # Removes the data affected by the cone of influence
        for i, col in enumerate(self.power.T):
            col_num = len(col) - i
            coi_start_index = min(range(len(self.freqs)),
                                  key=lambda i: abs(self.freqs[i] - self.coi[col_num]))
            self.power[:coi_start_index, col_num] = np.zeros(coi_start_index)        

    def remove_sheath(self):
        #   Removes all instinces of data inside the sheath
        sheath_windows_df = get_sheath_intervals('/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_v3.dat')
        sheath_window = False
        for index, row in sheath_windows_df.iterrows():
            if (self.time_series.min() < row.START < self.time_series.max()) or\
                (self.time_series.min() < row.END < self.time_series.max()):
                    mask = (self.time_series < row.START) | (self.time_series > row.END)
                    self.time_series = self.time_series[mask]
                    self.power = self.power[:, mask]
        
    def _peak_finding(self):

        mean_power = 2 * np.mean(self.power)
        peak_matrix = np.ma.masked_less_equal(self.power, mean_power).filled(fill_value=0)
        pws_per_freq = pd.DataFrame()
        for (freq, row) in zip(self.freqs, peak_matrix):
            peaks, _ = signal.find_peaks(row)
            prominence = signal.peak_prominences(row, peaks)
            power_arr = np.array([])
            for (peak, left, right) in zip(peaks, prominence[1], prominence[2]):
                peak_mean_power = np.mean(row[left: right])
                power_arr = np.append(power_arr, peak_mean_power)
            pws_per_freq = pd.concat([pws_per_freq, pd.DataFrame({freq: power_arr})],
                                     axis=1)

        max_mean_pwr = pws_per_freq.max().max()
        min_mean_pwr = pws_per_freq.min().min()
        self.freq_peaks_hist = np.array([])
        for freq, power_arr in pws_per_freq.items():
            for value in power_arr.dropna():
                weighted = 1 + (((value - min_mean_pwr)*(100 - 1))
                                / (max_mean_pwr - min_mean_pwr))
                for _ in range(int(weighted)):
                    self.freq_peaks_hist = np.append(
                        self.freq_peaks_hist, freq)
        self.peaks_found = True
        self.peak_matrix = peak_matrix

    def psd_calc(self):
        """Calculate the power spectral density of the cwt matrix by integrating along the time axis.
        """        
        self.psd = np.zeros(len(self.freqs))
        for i in range(0, len(self.freqs)):
            self.psd[i] = (2/len(self.freqs)) * integrate.trapz(self.power[i, :],
                                                                range(0, len(self.power[i,:])))
            
    def cwt_plot(self, axes, mark_coi=True, remove_coi=True, title=None, colorbar=True,
                 xlabel=None, time_per_major='12H', time_per_minor='1H',
                 tick_labels_format='%m-%d %H', x_ticks_labeled=True, **kwargs):

        if remove_coi:
            self.remove_coi()
        
        vmin = np.percentile(np.nan_to_num(self.power), 10)
        vmax = np.max(np.nan_to_num(self.power))
        t = mdates.date2num(self.time_series)
        plot_class = PlotClass(axes)
        plot_class.colormesh(t, self.freqs, self.power, ylabel='Frequency (Hz)', xlabel=xlabel,
                             title=title, color_bar=colorbar, norm=LogNorm(vmin=vmin, vmax=vmax),
                             cmap='jet', **kwargs)
        if mark_coi:
            plot_class.plot(self.time_series, self.coi, linestyle='--', color='black')
        axes.set_yscale('log')
        plot_class.xaxis_datetime_tick_labels(x_ticks_labeled)
        

    def peaks_plot(self, axes, plot_title=False, xlabel=True,
                   time_per_major='12H', time_per_minor='1H',
                   tick_labels_format='%m-%d %H', x_ticks_labeled=True, **kwargs):

        if not self.peaks_found:
            self._peak_finding()

        t = mdates.date2num(self.time_series)
        plot_class = PlotClass(axes)
        try:
            plot_class.colormesh(t, self.freqs, self.power, xlabel=xlabel, title=plot_title,
                            norm=LogNorm(),cmap='jet', **kwargs)
        except:
            plot_class.colormesh(t, self.freqs, self.power, xlabel=xlabel, title=plot_title,
                            cmap='jet', **kwargs)
        axes.set_yscale('log')
        plot_class.xaxis_datetime_tick_labels(x_ticks_labeled)

    def peaks_hist(self, axes, min_frequency=1/(20*60*60), max_frequency=1/(60*60),
                   freq_per_bin=1, x_units='min'):

        if not self.peaks_found:
            self._peak_finding(min_frequency, max_frequency)

        units_switch = {'sec': 1,
                        'min': 60,
                        'hour': 3600,
                        'day': 86400}
        bin_num = round(len(self.peak_freq_range) / freq_per_bin)
        axes.hist(self.freq_peaks_hist, bins=bin_num)
        axes.tick_params(axis='x', labelsize='small')
        axes.set_xticks(np.linspace(min_frequency, max_frequency, 10))
        axes.set_xticklabels(np.round(
            1/(np.linspace(min_frequency, max_frequency, 10) * units_switch[x_units]), 1
            ))
        axes.set_xlabel(f'Time ({x_units})')
        axes.set_ylabel('Weighted Peaks')
        axes.set_yscale('log')
        
    def psd_plot(self, axes, x_units, ylabel=None,
                 title=None, **kwargs):
        
        self.psd_calc()
        
        plot_class = PlotClass(axes, f'Time({x_units})', ylabel, title)
        plot_class.plot(self.freqs, self.psd, xlabel=f'Time ({x_units})', ylabel='Power',
                        color='blue', **kwargs)
        axes.set_yscale('log')
        
        units_switch = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}
        axes.set_xticks(np.linspace(self.freqs[0], self.freqs[-1], 10))
        axes.set_xticklabels(np.round(
                    1/(np.linspace(self.freqs[0], self.freqs[-1], 10) 
                       * units_switch[x_units.lower()]), 1))

    def calc_freq_bandpower(self):
        
        self.remove_coi()
        bandpower = np.zeros(len(self.time_series), dtype='float')
        for i in range(len(bandpower)):
            avg = integrate.trapz(self.power[:, i], self.freqs)
            bandpower[i] = avg

        return bandpower

class Turbulence(MagData):
    def __init__(self, start_iso, end_iso, dt, window_size=60, interval=30,
                 data_folder='/data/juno_spacecraft/data/fgm', instrument=['fgm_jno', 'r1s'],
                 wave_resolution=6, mother_wavelet=wavelet.Morlet):
        MagData.__init__(self, start_iso, end_iso, data_folder, instrument)
        self.mean_field_align(window_size)
        self.dt = dt
        self.interval = interval
        self.m = 23/6.0229e26
        self.z = 1.6
        self.q = 1.6e-19
        self.q_mhd = pd.Series(np.array([]), index=pd.DatetimeIndex([]))
        self.q_kaw = pd.Series(np.array([]), index=pd.DatetimeIndex([]))
        self.q_data = pd.Series(np.array([]), index=pd.DatetimeIndex([]))
        self._get_q()

    def _gyro(self, bx, by, bz, m, z, q):
        """finds a gyrofreqency given magnetosphere properties.  \n All inputs must be in
        fundamental units (T, m, etc.) \n Returns scalar qyrofrequency corresponding to given
        range of B"""
        mean_b = np.mean(np.sqrt(bx**2 + by**2 + bz**2))
        gyrofreq = (z*q/m)*(mean_b/(2*np.pi))
        return gyrofreq
    
    def _psd(self, cwt_power,freq,fs):
            """Finds PSD as per Tao et. al. 2015 given a morlet wavelet transform, frequency 
            range, the signal, and sampling frequency. \n Outputs an array of length of signal"""
            psd = np.zeros(len(freq))
            for i in range(0, len(freq)):
                psd[i] = (2/len(freq))*(sum(cwt_power[i, :]))
            return psd
                
    def _freqrange(self, f, gyro, psd):
        """Finds ranges of freqencies for MHD and KAW scales.\n  
        Inputs: f is frequency range for PSD, and gyro is the gyrofreqency for given domain. \n
        Returns the two frequency arrays and indices (tuples) for the two arrays. \n b1
        corresponds to MHD, b2 to KAW, and b3 to all real points of freqency range."""
        
        b1 = np.where((f>3E-3) & (f<(gyro)) & (psd>0))                                       #MHD range
        freq_mhd = f[b1]
        
        b2 = np.where((f>(gyro*1.5)) & (f<0.1))                                    #KAW range
        freq_kaw = f[b2]
        
        b3 = np.where((f>0) & (f<0.5))                                               #range for all real frequency
        return freq_mhd, freq_kaw, b1, b2, b3 
    
    def _q_calc(self, psd_perp, freq, bx, by, bz, b1, b2, m):
        """Takes PSD of perpendicular component and other parameters to find q MHD 
        and q KAW.  \n Every parameter must be in base units (T, kg, m, etc).  \n Empirical
        parameters are subject to adjustment below.  \n Outputs ranges of q MHD and q KAW 
        over freqency domains, according to b1 and b2 (respectively MHD and KAW freqency 
        domains. \n MAG vector components used only to find theta for k perp.  """
        delta_b_perp3 = (psd_perp*freq)**(3/2)                                                       #these parameters subject to change over spatial domain
        v_rel = 300e3
        n_density = 0.1*(100**3)
        density = m*n_density
        mu_0 = np.pi*4*1e-7
        kperp = (2*np.pi*freq)/(v_rel*np.sin(np.pi/2))  # currently just assumes v rel and B are perpendicular
        rho_i = 1e7 # parameter subject to change, currently an estimation from Tao et al
        qkaw = (0.5*(delta_b_perp3[b2])*kperp[b2]/np.sqrt(mu_0**3*density))*(1+kperp[b2]**2*rho_i**2)**0.5*(1+(1/(1+kperp[b2]**2*rho_i**2))*(1/(1+1.25*kperp[b2]**2*rho_i**2))**2)
        qmhd = (delta_b_perp3[b1])*kperp[b1]/(np.sqrt((mu_0**3)*density))
        
        return qmhd, qkaw
    
    def _get_q(self):
        
        mean_q_mhd = np.array([])
        mean_q_kaw = np.array([])
        self.q_kaw_slopes = np.array([])
        self.q_mhd_slopes = np.array([])
        num_intervals = np.ceil((self.data_df.index.max()
                         - self.data_df.index.min())/timedelta(minutes=self.interval))
        
        for i in range(int(num_intervals)):
            start_index = (self.data_df.index[0]
                            + timedelta(minutes=i*self.interval)).isoformat()
            end_index = (self.data_df.index[0]
                            + timedelta(minutes=(i+1)*self.interval)).isoformat()
            interval_time_series = self.data_df[start_index: end_index].index.to_pydatetime()
            if len(interval_time_series) < 1800:
                continue
            interval_bx = self.data_df.BX[start_index: end_index].to_numpy()
            interval_by = self.data_df.BY[start_index: end_index].to_numpy()
            interval_bz = self.data_df.BZ[start_index: end_index].to_numpy()
            interval_perp1 = self.data_df.B_PERP1[start_index: end_index].to_numpy()
            interval_perp2 = self.data_df.B_PERP2[start_index: end_index].to_numpy()

            fs = int(1/self.dt)
            
            cwt_perp1 = CWTData(interval_time_series,
                                interval_perp1,
                                self.dt)
            cwt_perp1.remove_coi()
            cwt_perp1.psd_calc()
            psd_perp1 = cwt_perp1.psd
            # psd_perp1 = self._psd(cwt_perp1.power, cwt_perp1.freqs, fs)
            
            cwt_perp2 = CWTData(interval_time_series,
                                interval_perp2,
                                self.dt)
            cwt_perp2.remove_coi()
            cwt_perp2.psd_calc()
            psd_perp2 = cwt_perp2.psd
            # psd_perp2 = self._psd(cwt_perp2.power, cwt_perp2.freqs, fs)

            psd_perp = (psd_perp1 + psd_perp2)*1e-18
            gyro_freq = self._gyro(interval_bx*1e-9, interval_by*1e-9, interval_bz*1e-9,
                                    self.m, self.z, self.q)
            freq_mhd, freq_kaw, b1, b2, b3 = self._freqrange(cwt_perp1.freqs, gyro_freq, psd_perp)
            q_mhd, q_kaw = self._q_calc(psd_perp, cwt_perp1.freqs,
                                interval_bx*1e-9, interval_by*1e-9, interval_bz*1e-9,
                                b1, b2, self.m)
            mean_q_mhd = np.append(mean_q_mhd, np.mean(q_mhd))
            mean_q_kaw = np.append(mean_q_kaw, np.mean(q_kaw))
            if len(q_mhd) == 0 or len(q_kaw) == 0:                          #check that there is a KAW or MHD scale on frequency range
                    pass
                
            else:
                r = LinearRegression()
                r.fit(np.reshape(np.log10(freq_mhd), (-1,1)),
                      np.reshape(np.log10(psd_perp[b1]),(-1,1)))
                q_mhd_slope = r.coef_[0]
                self.q_mhd_slopes = np.append(self.q_mhd_slopes, q_mhd_slope)
                r.fit(np.reshape(np.log10(freq_kaw),(-1,1)),
                      np.reshape(np.log10(psd_perp[b2]),(-1,1)))
                q_kaw_slope = r.coef_[0]
                self.q_kaw_slopes = np.append(self.q_kaw_slopes, q_kaw_slope)
        
        mean_q = (mean_q_mhd + mean_q_kaw)/2
        time = pd.date_range(self.data_df.index[0].isoformat(),
                             self.data_df.index[-1].isoformat(),
                             periods=len(mean_q))
        self.q_data = pd.DataFrame({'q': mean_q,
                                    'q_kaw': mean_q_kaw,
                                    'q_mhd': mean_q_mhd,},
                                   index=time)
        
    def q_plot(self, axes, start=None, end=None, title=None, xlabel=None, x_ticks_labeled=False):
        
        if start & end:
            plot_data = self.q_data[start.isoformat(): end.isoformat()]
        else:
            plot_data = self.q_data
        
        
        for i in range(0, len(plot_data.index)-1):
            axes.plot((plot_data.index[i], plot_data.index[i+1]),
                      (plot_data.to_numpy()[i], plot_data.to_numpy()[i]),
                      color='blue')
        axes.set_yscale('log')
        axes.set_ylabel('Q(W/m^3)')
        axes.set_xlabel(xlabel)
        axes.set_title(title)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=20)
        formatter = mdates.ConciseDateFormatter(locator)
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(formatter)
        if not x_ticks_labeled:
            axes.set_xticklabels([])
        axes.set_xlim(plot_data.index[0], plot_data.index[-1])

class PDS3Label(): 
    """Class for reading and parsing PDS3 labels, this will only work with labels that contain the comma seperated comment keys. e.g. /*RJW, Name, Format, dimnum, size dim 1, size dim 2,...*/\n
    returns a dictionary """
    def __init__(self,labelFile):
        self.label = labelFile
        self.dataNames = ['DIM0_UTC','PACKET_SPECIES','DATA','DIM1_E','SC_POS_LAT','SC_POS_R'] #All the object names you want to find info on from the .lbl file
        self.dataNameDict = {} #Initialization of a dictionary that will index other dictionaries based on the data name
        self.getLabelData() #Automatically calls the function to get data from the label 
        

    def getLabelData(self):
        byteSizeRef = {'c':1,'b':1,'B':1,'?':1,'h':2,'H':2,'i':4,'I':4,'l':4,'L':4,'q':8,'Q':8,'f':4,'d':8} #Size of each binary format character in bytes to help find starting byte
        byteNum = 0
        with open(self.label) as f:
            line = f.readline()
            while line != '':   #Each line is read through in the label
                line = f.readline()
                if 'FILE_RECORDS' in line:
                    self.rows = int(line[12:].strip().lstrip('=').strip())
                if line[:6] == '/* RJW':    #If a comment key is found parsing it begins
                    line = line.strip().strip('/* RJW,').strip().split(', ')    #The key is split up into a list removing the RJW
                    if line[0] == 'BYTES_PER_RECORD':
                        self.bytesPerRow = int(line[1])
                        continue
                    if len(line) > 2:
                        if line[0] in self.dataNames:   #If the line read is a comment key for one of the objects defined above the data will be put into a dictionary
                            self.dataNameDict[line[0]] = {'FORMAT':line[1],'NUM_DIMS':line[2],'START_BYTE':byteNum}
                            for i in range(int(line[2])):
                                self.dataNameDict[line[0]]['DIM'+str(i+1)] = int(line[i+3])
                        byteNum += np.prod([int(i) for i in line[3:]])*byteSizeRef[line[1]] #Using the above dictionary the total size of the object is found to find the ending byte
                        if line[0] in self.dataNames:
                            self.dataNameDict[line[0]]['END_BYTE'] = byteNum
                    
        return self.dataNameDict #The dictionary is returned


    
class JadeData():
    """Class for reading and getting data from a list of .dat file from the get files function provides.\n
    Datafile must be a single .dat file.\n
    Start time must be in UTC e.g. '2017-03-09T00:00:00.000'.\n
    End time must be in UTC e.g. '2017-03-09T00:00:00.000'.\n
    """
    def __init__(self,dataFile,startTime,endTime):
        self.dataFileList = dataFile
        self.startTime = datetime.fromisoformat(startTime) #Converted to datetime object for easier date manipulation
        self.endTime = datetime.fromisoformat(endTime)
        self.dataDict = {}
        self.ion_df = pd.DataFrame({'DATA':[]})
        self.ion_dims = None
        self.elec_df = pd.DataFrame({'ELC_DATA':[]})
        self.elec_dims = None
        

    def getIonData(self):   
        for dataFile in self.dataFileList:
            labelPath = dataFile.rstrip('.DAT') + '.LBL'    #All .dat files should come with an accompanying .lbl file
            label = PDS3Label(labelPath)    #The label file is parsed for the data needed
            rows = label.rows #All LRS jade data has 8640 rows of data per file
            species = 3 #The ion species interested in as defined in the label
            with open(dataFile, 'rb') as f:
                for _ in range(rows):
                    data = f.read(label.bytesPerRow)    
                    
                    
                    timeData = label.dataNameDict['DIM0_UTC']   #Label data for the time stamp
                    startByte = timeData['START_BYTE']  #Byte where the time stamp starts
                    endByte = timeData['END_BYTE']  #Byte where the time stamp ends
                    dataSlice = data[startByte:endByte] #The slice of data that contains the time stamp
                    dateTimeStamp = datetime.strptime(str(dataSlice,'ascii'), '%Y-%jT%H:%M:%S.%f').replace(microsecond=0)  #The time stamp is converted from DOY format to a datetime object
                    dateStamp = str(dateTimeStamp.date())   #A string of the day date to be used as the main organizational key in the data dictionary
                    time = dateTimeStamp.time() #The time in hours to microseconds for the row
                    timeStamp = time.hour + time.minute/60 + time.second/3600   #Convert the time to decimal hours

                    if dateStamp in self.dataDict:  #Check if a entry for the date already exists in the data dictionary
                        pass
                    else:
                        self.dataDict[dateStamp] = {}
                        
                    if dateTimeStamp > self.endTime:    #If the desired end date has been passed the function ends
                            f.close()   
                            return
                    speciesObjectData = label.dataNameDict['PACKET_SPECIES']    #The species data from teh label is pulled
                    startByte = speciesObjectData['START_BYTE']
                    endByte = speciesObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte]
                    ionSpecies = struct.unpack(speciesObjectData['FORMAT']*speciesObjectData['DIM1'],dataSlice)[0] #Species type for the row is found

                    dataObjectData = label.dataNameDict['DIM1_E'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [row[0] for row in temp]  #Each rows average is found to have one column 

                    if self.ion_dims is None:
                        self.ion_dims = dataArray

                    if ionSpecies == species:   #If the species for the row is the desired species continue finding data
                                                    
                        dataObjectData = label.dataNameDict['DATA'] #Label data for the data is found 
                        startByte = dataObjectData['START_BYTE']
                        endByte = dataObjectData['END_BYTE']
                        dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                        temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                        temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                        dataArray = [np.mean(row) for row in temp]  #Each rows average is found to have one column 
                        
                        temp_df = pd.DataFrame({dateTimeStamp: (dataArray)}).transpose()
                        self.ion_df = self.ion_df.append(temp_df)
                        
            f.close()

    def getElecData(self):
        for dataFile in self.dataFileList:
            labelPath = dataFile.rstrip('.DAT') + '.LBL'    #All .dat files should come with an accompanying .lbl file
            label = PDS3Label(labelPath)    #The label file is parsed for the data needed
            rows = label.rows #All LRS jade data has 8640 rows of data per file
            with open(dataFile, 'rb') as f:
                for _ in range(rows):
                    data = f.read(label.bytesPerRow)    
                    
                    timeData = label.dataNameDict['DIM0_UTC']   #Label data for the time stamp
                    startByte = timeData['START_BYTE']  #Byte where the time stamp starts
                    endByte = timeData['END_BYTE']  #Byte where the time stamp ends
                    dataSlice = data[startByte:endByte] #The slice of data that contains the time stamp
                    dateTimeStamp = datetime.strptime(str(dataSlice,'ascii'),'%Y-%jT%H:%M:%S.%f')  #The time stamp is converted from DOY format to a datetime object
                    dateStamp = str(dateTimeStamp.date())   #A string of the day date to be used as the main organizational key in the data dictionary
                    time = dateTimeStamp.time() #The time in hours to microseconds for the row
                    timeStamp = time.hour + time.minute/60 + time.second/3600   #Convert the time to decimal hours

                    if dateStamp in self.dataDict:  #Check if a entry for the date already exists in the data dictionary
                        pass
                    else:
                        self.dataDict[dateStamp] = {}
                        
                    if dateTimeStamp > self.endTime:    #If the desired end date has been passed the function ends
                            f.close()   
                            return 
                    
                    dataObjectData = label.dataNameDict['DATA'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [np.mean(row) for row in temp]  #Each rows average is found to have one column 
                    temp_df = pd.DataFrame({dateTimeStamp: (dataArray)}).transpose()
                    self.elec_df = self.elec_df.append(temp_df)

                    dataObjectData = label.dataNameDict['DIM1_E'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [row[0] for row in temp]  #Each rows average is found to have one column
                    #dataArray = [np.mean(row) for row in temp]  #Each rows average is found to have one column 
                    if self.elec_dims is None:
                        self.elec_dims = dataArray

    
class JadClass(bc_ids):
    def __init__(self,timeStart, timeEnd, species='ION'):  #type must be either "ION" or "ELC"
        self.t = 0.0
        self.t_e = 0.0
        self.energy_scale = 0.0
        #self.energy_scale_elec = 0.0
        self.jad_arr = 0.0
        #self.jad_e_arr = 0.0
        self.jad_mean = 0.0
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.z_cent = 0.0
        self.R = 0.0
        self.bc_df = 0.0
        self.bc_id = 0.0
        self.species = species

        #self.read_ion_data()
        self.read_data()
        #x,y,z = self.sys_3_data()
        self.bc_df = self.get_mp_bc()
        self.get_bc_mask()
        
    def read_data(self):      
        dataFolder = pathlib.Path('/data/juno_spacecraft/data/jad')
        datFiles = _get_files(self.timeStart,self.timeEnd,'.DAT',dataFolder,'JAD_L30_LRS_'+self.species+'_ANY_CNT') 
        print('getting jade data....'+self.species)
        if (self.species == 'ION'):
            jadeIon = JadeData(datFiles,self.timeStart,self.timeEnd)
            jadeIon.getIonData()
            jadeIonData = jadeIon.dataDict
            jadeIonData = jadeIon.ion_df
            
            self.jad_mean = []
            self.t = jadeIon.ion_df.index
            self.jad_arr = jadeIon.ion_df.to_numpy().transpose()[: -1,:]
            print('jad_arr...',np.shape(self.jad_arr))
            self.energy_scale = np.array(jadeIon.ion_dims)/1000
            #plt.figure()
            #plt.pcolormesh(self.t,self.energy_scale,np.log(self.jad_arr),cmap='jet')
            #plt.yscale('log')
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

        if (self.species == 'ELC'):
            jadeElc = JadeData(datFiles,self.timeStart,self.timeEnd)
            jadeElc.getElecData()
            jadeElcData = jadeElc.dataDict
            jadeElcData = jadeElc.elec_df
            
            self.t = jadeElc.elec_df.index
            self.jad_arr = jadeElc.elec_df.to_numpy().transpose()[: -1,:]
            self.energy_scale = np.array(jadeElc.elec_dims)/1000
        print('jade data retrieved...')

        
    def read_elec_data(self):      
        dataFolder = pathlib.Path('/data/juno_spacecraft/data/jad')
        datFiles = _get_files(self.timeStart,self.timeEnd,'.DAT',dataFolder,'JAD_L30_LRS_ELC_ANY_CNT') 
        jadeElc = JadeData(datFiles,self.timeStart,self.timeEnd)
        print('getting electron data....')
        jadeElc.getElecData()
        print('electron data retrieved...')
        #plt.figure()
        #if date in jadeIon.dataDict.keys(): #Ion spectrogram portion
        jadeElcData = jadeElc.dataDict
        jadeElcData = jadeElc.elec_df
        
        self.t = jadeElc.elec_df.index
        self.jad_e_arr = jadeElc.elec_df.to_numpy().transpose()[: -1,:]
        self.energy_scale_elec = np.array(jadeElc.elec_dims)/1000
        #plt.figure()
        #plt.pcolormesh(self.t,self.energy_scale_elec,np.log(self.jad_e_arr),cmap='jet')
        #plt.yscale('log')
        #plt.show()
        #sz = self.jad_i_arr.shape
        #for i in range(sz[0]):
        #    self.jad_mean.append(self.jad_i_arr[i,:-2].mean())
        #    #self.jad_max.append(self.jad_arr[i,:-2].max())
        #    #plt.figure()
        #    #plt.plot(jad_tm,jad_mean)
        #    #plt.plot(jad_tm,jad_max)
        #    #plt.show()
        #self.jad_mean = np.array(self.jad_mean)

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


"""
if __name__ == '__main__':
    start = '2016-07-31T00:00:00'
    end = '2020-11-06T12:00:00'
    
    lat_df = time_in_lat_window(start, end, 30)
    for index, row in lat_df.iterrows():
        start_datetime = row['START']
        end_datetime = row['END']
        file = f'q_data_{start_datetime.date()}-{end_datetime.date()}.pickle'
    
        turb = Turbulence(start_datetime.isoformat(),
                          end_datetime.isoformat(),
                          1, 30, 30)
        file_path = f'/home/aschok/Documents/data/heating_data_+-30/{file}'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(turb.q_data, pickle_file)
            print(f'Saved data from {start_datetime} to {end_datetime}')
            pickle_file.close()
        
        file = f'q_slopes_{start_datetime.date()}-{end_datetime.date()}.pickle'
        file_path = f'/home/aschok/Documents/data/heating_data_+-30/{file}'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump({'q_kaw_slopes': turb.q_kaw_slopes,
                         'q_mhd_slopes': turb.q_mhd_slopes,
                         'time': turb.q_data.index}
                         , pickle_file)
            print(f'Saved data from {start_datetime} to {end_datetime}')
            pickle_file.close()
"""

class HuscherClass:
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
        return n_dens,n_sig,n_r,n_df
        
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
