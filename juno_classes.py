
import re
import pickle
import struct
from datetime import datetime, timedelta
from os import fsdecode

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

from juno_functions import _get_files, time_in_lat_window, get_sheath_intervals


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

    def __init__(self, start_time, end_time, data_folder='/data/juno_spacecraft/data/fgm',
                 instrument=['fgm_jno', 'r1s']):
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
        self.data_df = pd.DataFrame()
        self._get_data()

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
        self.elec_df = pd.DataFrame()
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
                    
                    temp_df = pd.DataFrame({dateTimeStamp: [np.log(dataArray)]}).transpose()
                    self.elec_df = self.elec_df.append(temp_df)                    

                    dataObjectData = label.dataNameDict['DIM1_E'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [row[0] for row in temp]  #Each rows average is found to have one column 
                    if self.elec_dims is None:
                        self.elec_dims = dataArray

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
