#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:55:16 2021

@author: aschok
"""

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import spiceypy as spice
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, ScalarFormatter)
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from space_datamodule import *
import pycwt as wavelet
from astropy.io import ascii


def mag_cwt_plot(start_time, end_time, save_location,
               window_size=24, days_per_graph=8,
               max_frequency=1/(60*60), min_frequency=1/(20*60*60)):
    # A function which finds all the windows of time between start_time and end_time in which Juno is between +- target latitude
    lat_windows_df = time_in_lat_window(start_time, end_time, target_latitude = 10)
    cum_hist = []
    for index, row in lat_windows_df.iterrows():
        start_datetime = row['START']
        end_datetime = row['END']
        total_hist_save = f'Cumulative histogram from {start_datetime.date()} to {end_datetime.date()}'
        mag_data = MFAData(start_datetime, end_datetime).mag_data
        num_graphs = round((mag_data.index[-1] - mag_data.index[0])/pd.Timedelta(days = days_per_graph)) + 1 
        end_datetime = mag_data.index[0]
        total_peaks_hist = []
        # Constants
        omega0 = 6  # change this parameter to adjust "resolution" of waveform
        dt = 60
        Fs = 1 / dt
        
        total_peaks_hist = []
        for i in range(0,num_graphs):
            start_datetime = end_datetime 
            end_datetime = start_datetime + pd.Timedelta(days = days_per_graph)
            graph_data = mag_data[start_datetime.isoformat() : end_datetime.isoformat()]
            end_datetime = graph_data.index[-1]
            save_name = f'Peaks {start_datetime.date()}_{end_datetime.date()}'
            graph_data = graph_data.fillna(0)
            if len(graph_data) <= 2 or (end_datetime - start_datetime) < timedelta(hours=4):
                break
            
            # Graph area is created
# =============================================================================
#             grid = gs.GridSpec(3, 2, wspace=0.265)
#             fig = plt.figure(figsize = (20,10))
#             fig.suptitle(f'{start_datetime.date()} to {end_datetime.date()}', fontsize=20)
# =============================================================================
            
            # Raw mag data is graphed
# =============================================================================
#             fig, ax0 = plt.subplots(figsize=(10,5))
#             ax0.plot(graph_data.index, graph_data.BX, label='BX', linewidth=0.5)
#             ax0.plot(graph_data.index, graph_data.BY, label='BY', linewidth=0.5)
#             ax0.plot(graph_data.index, graph_data.BZ, label='BZ', linewidth=0.5)
#             ax0.plot(graph_data.index, graph_data.B, label='B', color='black', linewidth=0.5)
#             ax0.plot(graph_data.index, -graph_data.B, color='black', linewidth=0.5)
#             ax0.set_xlim([graph_data.index[0], graph_data.index[-1]])
#             fmt_hr_major = mdates.HourLocator(byhour=10)
#             ax0.xaxis.set_major_locator(fmt_hr_major)
#             fmt_hr_minor = mdates.HourLocator(byhour=1)
#             ax0.xaxis.set_minor_locator(fmt_hr_minor)
#             ax0.set_xlabel('Magnetic field (nT)')
#             ax0.set_ylabel('Time')
#             ax0.legend()
# =============================================================================
            
# =============================================================================
#             fig, ax0 = plt.subplots(figsize=(10,5))
#             ax0.plot(graph_data.index, signal.detrend(graph_data.B_PERP2), linewidth=0.5)
#             ax0.set_xlim([graph_data.index[0], graph_data.index[-1]])
#             fmt_hr_major = mdates.HourLocator(byhour=10)
#             ax0.xaxis.set_major_locator(fmt_hr_major)
#             fmt_hr_minor = mdates.HourLocator(byhour=1)
#             ax0.xaxis.set_minor_locator(fmt_hr_minor)
#             ax0.set_xlabel('$\Delta B_{perp2}$ of MFA magnetic field (nT)')
#             ax0.set_ylabel('Time')
# =============================================================================
            
            N = len(graph_data)
            f = np.arange(0, N / 2 + 1) * Fs / N
            t = np.arange(0, N) * dt
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal.detrend(graph_data.B_PERP2), dt, wavelet.Morlet(6), freqs=f[1:])
            power = abs(wave)**2
            coi = coi**-1
            
            # Raw CWT data graphed
# =============================================================================
#             fig, ax1 = plt.subplots(figsize=(10,5))
#             cwt = ax1.pcolormesh(t, f, power, cmap = 'jet', norm = LogNorm(), vmin = np.percentile(power, 10))
#             ax1.plot(t, coi, linestyle='--', color='black')
#             ax1.set_yscale('log')
#             ax1.set_ylim(f[1], f[-1])
#             ax1.set_ylabel('Frequency (Hz)')
#             ax1.set_xlabel('Time')
#             ax1.set_xticks(np.arange(0, t[-1], 60*60*12))
#             ax1.set_xticks(np.arange(0, t[-1], 60*60), minor = True)
#             twelve_hr_labels = pd.date_range(graph_data.index[0].isoformat(),
#                                              graph_data.index[-1].isoformat(),
#                                              freq='12H').strftime('%m-%d %H')
#             ax1.set_xticklabels(twelve_hr_labels)
#             axins = inset_axes(ax1,
#                        width="2%",  
#                        height="100%",  
#                        loc='center right',
#                        bbox_to_anchor=(0.04, 0, 1, 1),
#                        bbox_transform=ax1.transAxes,
#                        borderpad=0,
#                        )
#             cbr = plt.colorbar(cwt, cax=axins)
#             cbr.set_label('Magnitude' ,rotation=270, labelpad=10, size=9)
#             ax1.hlines([1/(10*60*60) , 1/(1*60*60) , 1/(10*60)], 0, (N-1)*60, colors='white', linestyles='dashed')
#             trans = transforms.blended_transform_factory(ax1.transAxes, ax1.transData)
#             ax1.text(-0.037, 1/(10*60), r'$\frac{1}{10 Min}$', fontsize='medium', transform=trans)
#             ax1.text(-0.037, 1/(1*60*60), r'$\frac{1}{60 Min}$', fontsize='medium', transform=trans)
#             ax1.text(-0.037, 1/(10*60*60), r'$\frac{1}{10 Hr}$', fontsize='medium', transform=trans)
# =============================================================================
            
            # Begin peak finding
            freq_peaks_array = []
            min_freq_index = min(range(len(f)), key = lambda i: abs(f[i] - min_frequency))
            max_freq_index = min(range(len(f)), key = lambda i:  abs(f[i] - max_frequency))
            freq_range = f[min_freq_index : max_freq_index]
            power = power[min_freq_index : max_freq_index, : ]
            for i, col in enumerate(power.T):
                col_num = len(col) - i
                coi_start_index = min(range(len(freq_range)), key = lambda i : abs(freq_range[i] - coi[col_num]))
                power[:coi_start_index,col_num] = np.zeros(coi_start_index)
            mean_power = np.mean(power)
            peak_matrix = np.ma.masked_less_equal(power, mean_power).filled(fill_value=0)
            
            max_peak = np.max(peak_matrix[signal.argrelmax(peak_matrix, axis=1)])
            max_row, max_col = np.where(peak_matrix == max_peak)
            min_peak = np.min(peak_matrix[signal.argrelmax(peak_matrix, axis=1)])
            min_row, min_col = np.where(peak_matrix == min_peak)
            for i, (row_num, col_num) in enumerate(zip([max_row[0], min_row[0]], [max_col[0], min_col[0]])):
                row = peak_matrix[row_num,]
                local_max = signal.argrelmax(row)[0]
                local_min = signal.argrelmin(row)[0]
                left = col_num
                right = col_num
                while True:
                    if left == 0 or row[left] == 0 or left in local_min:
                        break
                    else:
                        left -= 1

                while True:
                    if right == len(row[:-1]) or row[right] == 0 or right in local_min:
                        break
                    else:
                        right += 1

                if i == 0:
                    max_mean_power = np.mean(row[left : right])
                elif i == 1:
                    min_mean_power = np.mean(row[left : right])

            norm_matrix = 1 + ((peak_matrix - min_mean_power)*(50 - 1))/(max_mean_power - min_mean_power)
            for (freq, row) in zip(freq_range, norm_matrix):
                local_max = signal.argrelmax(row)[0]
                local_min = signal.argrelmin(row)[0]
                for max_loc in local_max:
                    left = max_loc
                    right = max_loc
                    while True:
                        if left == 0 or row[left] == 0 or left in local_min:
                            break
                        else:
                            left -= 1

                    while True:
                        if right == len(row[:-1]) or row[right] == 0 or right in local_min:
                            break
                        else:
                            right += 1

                    mean_peak_pow = np.mean(row[left: right])
                    for i in range(int(mean_peak_pow)+1):
                        freq_peaks_array = np.append(freq_peaks_array, freq)
                        total_peaks_hist = np.append(total_peaks_hist, freq)
                        cum_hist = np.append(cum_hist, freq)
            
            # Peak CWT plot
            fig, ax2 = plt.subplots(figsize=(10, 5))
            ax2.pcolormesh(t, freq_range, peak_matrix, cmap='jet', norm=LogNorm())
            ax2.set_yscale('log')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_xticks(np.arange(0, t[-1], 60*60*12))
            ax2.set_xticks(np.arange(0, t[-1], 60*60), minor = True)
            twelve_hr_labels = pd.date_range(graph_data.index[0].isoformat(),
                                             graph_data.index[-1].isoformat(),
                                             freq='12H').strftime('%m-%d %H')
            ax2.set_xticklabels(twelve_hr_labels)
            ax2.set_xlabel('Time (Hr)')
            ax2.tick_params(axis='x', which='major', labelsize='medium')
            
            # Weighted peaks histograms
            fig, ax3 = plt.subplots(1, 1, figsize=(9, 10))
            freq_per_bin = 3
            bin_num = round(len(freq_range)/freq_per_bin)
            n, bins, _ = ax3.hist(freq_peaks_array, bins=bin_num)
            ax3.tick_params(axis='x', labelsize='medium')
            ax3.set_xlabel('Time (Hr)')
            ax3.set_ylabel('Peak Power')
            ax3.set_yscale('log')
            ax3.set_xticks(np.linspace(min_frequency, max_frequency, 8))
            ax3.set_xticklabels(np.round(1/(np.linspace(min_frequency, max_frequency, 8)*3600), 1))
            freq_start_vert_lines = 1/(10*60*60)
            
            # Weighted peaks histograms
            fig, ax3 = plt.subplots(1, 1, figsize=(9, 10))
            freq_per_bin = 1
            bin_num = round(len(freq_range)/freq_per_bin)
            n, bins, _ = ax3.hist(freq_peaks_array, bins=bin_num)
            ax3.tick_params(axis='x', labelsize='medium')
            ax3.set_xlabel('Time (Hr)')
            ax3.set_ylabel('Peak Power')
            ax3.set_yscale('log')
            ax3.set_xscale('log')
            ax3.set_xticks(np.linspace(min_frequency, max_frequency, 8))
            ax3.set_xticklabels(np.round(1/(np.linspace(min_frequency, max_frequency, 8)*3600), 1))
               
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(cum_hist, bins=round(len(freq_range)/2))
    ax.tick_params(axis='x', labelsize='medium')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('log(weighted peaks)')
    ax.set_yscale('log')
    
            
if __name__ == '__main__':
    start_time = '2016-11-26T00:00:00'
    end_time = '2016-11-30T23:59:59'
    save_location = r'/home/aschok/Documents/figures/cwt/testing'
    mag_cwt_plot(start_time, end_time, save_location, days_per_graph=4)


            
