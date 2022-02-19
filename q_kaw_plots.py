import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import spiceypy as spice
from scipy.stats import binned_statistic
from datetime import datetime
import spiceypy as spice
from juno_functions import time_in_lat_window, find_orb_num

class QKAWData:
    def __init__(self, data_folder, minlat=-10, maxlat=0):
        self.data_folder = data_folder
        self.minlat = minlat
        self.maxlat = maxlat
        self._get_data()
        
    def _get_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
        local_time_array = np.array([])
        radial_array = np.array([])
        latitude_array = np.array([])
        q_kaw_array = np.array([])
        mean_q_kaw_array = np.array([])
        orb_start  = np.array([])
        orb_num_array = np.array([])
        
        #Using the crossing masterlist, finds date ranges when Juno is within the sheah to remove this data
        crossings_df = pd.read_csv('/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_v6.txt')
        crossings_df = crossings_df.drop('NOTES', axis=1)
        in_sheath = False
        sheath_windows_df = pd.DataFrame({'START': [], 'END': []}) 
        for index, row in crossings_df.iterrows():
            if row.BOUNDARYID.lower() == 'sheath':
                if not in_sheath:
                    start = datetime.fromisoformat(f'{row.DATE}T{row.TIME}')
                    in_sheath = True
                
            if row.BOUNDARYID.lower() == 'magnetosphere':
                if in_sheath:
                    end = datetime.fromisoformat(f'{row.DATE}T{row.TIME}')
                    in_sheath = False
                    sheath_windows_df = sheath_windows_df.append({'START':start,
                                                                'END': end},
                                                                ignore_index=True)

        # Data files are opened and the data is read in and cleaned
        r = re.compile('_data_')
        for root, dirs, files in os.walk(self.data_folder):
            for file_name in files:
                if r.search(file_name):
                    with open(os.path.join(root, file_name), 'rb') as data_file:
                        pkl = pickle.load(data_file)
                        
                        # Q values inside the sheath are removed
                        sheath_window = False
                        data = pkl
                        temp_data = pd.DataFrame()
                        for index, row in sheath_windows_df.iterrows():
                            if (pkl.index.min() < row.START < pkl.index.max()) or\
                            (pkl.index.min() < row.END < pkl.index.max()):
                                sheath_window = True
                                data = data[(data.index < row.START) | (data.index > row.END)]
                        if sheath_window:
                            pkl = data
                            
                        orb_num = find_orb_num(pkl.index.min())
                        et_array = [spice.utc2et(i) for i in pkl.index.strftime('%Y-%m-%dT%H:%M:%S')]
                        positions, lt = spice.spkpos('JUNO', et_array,
                                                    'JUNO_JSS', 'NONE', 'JUPITER')
                        x = positions.T[0]
                        y = positions.T[1]
                        z = positions.T[2]
                        rad = [np.sqrt(np.sum(np.power(vector, 2))) for vector in positions]
                        lat = np.arcsin(z / rad) * (180/np.pi)
                        long = np.arctan2(y, x) *(180/np.pi)
                        local_time = ((long + 180)*24/360)%24
                        
                        mask = (lat<=self.maxlat) & (lat>=self.minlat) & (np.isfinite(np.array(pkl.q_kaw)))
                        lat = np.array(lat)[mask]
                        rad = np.array(rad)[mask]
                        local_time = np.array(local_time)[mask]
                        q_kaw = np.array(pkl.q_kaw)[mask]
                        
                        local_time_array = np.append(local_time_array, np.mean(local_time))
                        radial_array = np.append(radial_array, np.array(rad)/69911)
                        latitude_array = np.append(latitude_array, lat)
                        q_kaw_array = np.append(q_kaw_array, q_kaw)
                        mean_q_kaw_array = np.append(mean_q_kaw_array, np.mean(q_kaw))
                        orb_num_array = np.append(orb_num_array, np.full(len(q_kaw), orb_num))
                        
        spice.kclear() 
        self.q_v_lt = pd.DataFrame({'local_time': local_time_array, 'q_kaw': mean_q_kaw_array})
        self.q_v_pos = pd.DataFrame({'orbit': orb_num_array, 'q_kaw': q_kaw_array,
                                     'radial': radial_array, 'latitude': latitude_array})

    
def q_plot_all_orbs():
    q_plotter = QKAWData('/home/aschok/Documents/data/heating_data_+-10')
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,8))
    
    q_v_lt_plot = q_plotter.q_v_lt
    ax1.scatter(q_v_lt_plot.local_time, q_v_lt_plot.q_kaw, color='blue')
    ax1.set_title('$q_{kaw}$ vs. Local Time')
    ax1.set_ylabel('$q_{kaw}$')
    ax1.set_xlabel('Local Time')
    ax1.set_yscale('log')
    
    q_v_pos_data = q_plotter.q_v_pos
    bins = np.arange(np.floor(np.min(q_v_pos_data.latitude)) - 0.5,
                    np.ceil(np.max(q_v_pos_data.latitude)) + 1.5)
    means, edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                        'mean', bins)
    bin_std, std_edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                            'std', bins)
    edges += 0.5
    ax2.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax2.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax2.set_xlabel('Latitude (deg)')
    ax2.set_ylabel('$q_{kaw}$')
    ax2.set_title(f'$q_{{kaw}}$ vs. Latitude')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    
    bins = np.arange(np.floor(np.min(q_v_pos_data.radial)) - 0.5,
                        np.ceil(np.max(q_v_pos_data.radial)) + 1.5)
    means, edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                        'mean', bins)
    bin_std, std_edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                            'std', bins)
    edges += 0.5
    ax3.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax3.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax3.set_xlabel('R ($R_j$)')
    ax3.set_ylabel('$q_{kaw}$')
    ax3.set_title('$q_{kaw}$ vs. Radial distance')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')
    ax3.set_xlim(edges[0], edges[-1])
    plt.tight_layout(h_pad=0.25)
    plt.show()
          
def q_plot_each_orb():
    q_plotter = QKAWData('/home/aschok/Documents/data/heating_data_+-10')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,8))
    q_v_lt_plot = q_plotter.q_v_lt
    ax1.scatter(q_v_lt_plot.local_time, q_v_lt_plot.q_kaw, color='blue')
    ax1.set_title('$q_{kaw}$ vs. Local Time')
    ax1.set_ylabel('$q_{kaw}$')
    ax1.set_xlabel('Local Time')
    ax1.set_yscale('log')
    
    q_v_pos_data = q_plotter.q_v_pos
    bins = np.arange(np.floor(np.min(q_v_pos_data.latitude)) - 0.5,
                    np.ceil(np.max(q_v_pos_data.latitude)) + 1.5)
    means, edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                        'mean', bins)
    bin_std, std_edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                            'std', bins)
    edges += 0.5
    ax2.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax2.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax2.set_xlabel('Latitude (deg)')
    ax2.set_ylabel('$q_{kaw}$')
    ax2.set_title(f'$q_{{kaw}}$ vs. Latitude')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    
    bins = np.arange(np.floor(np.min(q_v_pos_data.radial)) - 0.5,
                        np.ceil(np.max(q_v_pos_data.radial)) + 1.5)
    means, edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                        'mean', bins)
    bin_std, std_edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                            'std', bins)
    edges += 0.5
    ax3.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax3.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax3.set_xlabel('R ($R_j$)')
    ax3.set_ylabel('$q_{kaw}$')
    ax3.set_title('$q_{kaw}$ vs. Radial distance')
    ax3.set_yscale('log')
    ax3.set_xlim(edges[0], edges[-1])
    
    
    for orb in q_plotter.q_v_pos.orbit.unique():
        
        q_v_pos_data = q_plotter.q_v_pos[q_plotter.q_v_pos.orbit == orb]
        if np.max(q_v_pos_data.radial) < 85:
            continue
        
        if (np.isnan(np.min(q_v_pos_data.latitude))) or (np.isnan(np.max(q_v_pos_data.latitude))):
            continue
        
        bins = np.arange(np.floor(np.min(q_v_pos_data.latitude)) - 0.5,
                        np.ceil(np.max(q_v_pos_data.latitude)) + 1.5)
        means, edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                            'mean', bins)
        bin_std, std_edges, num = binned_statistic(q_v_pos_data.latitude, q_v_pos_data.q_kaw,
                                                'std', bins)
        edges += 0.5
        ax2.plot(edges[:-1], means, label=f'{orb}')
        ax2.set_xlabel('Latitude (deg)')
        ax2.set_ylabel('$q_{kaw}$')
        ax2.set_title(f'$q_{{kaw}}$ vs. Latitude')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')

        bins = np.arange(np.floor(np.min(q_v_pos_data.radial)) - 0.5,
                            np.ceil(np.max(q_v_pos_data.radial)) + 1.5)
        means, edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                            'mean', bins)
        bin_std, std_edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                                'std', bins)
        edges += 0.5
        ax3.plot(edges[:-1], means, label=f'{orb}')
        ax3.set_xlabel('R ($R_j$)')
        ax3.set_ylabel('$q_{kaw}$')
        ax3.set_title('$q_{kaw}$ vs. Radial distance')
        ax3.set_yscale('log')
        ax3.legend(loc='upper left')
        
    plt.show()
        
def orbits_plot():
    
    start = '2016-07-31T00:00:00'
    end = '2020-12-04T00:00:00'
    time_series = pd.date_range(start, end, freq='1H')
    for year in ['2016', '2017', '2018', '2019', '2020']:
            spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
    et_range = [spice.utc2et(i) for i in time_series.strftime('%Y-%m-%dT%H:%M:%S')]

    positions, lt = spice.spkpos('JUNO', et_range, 'JUNO_JSS', 'NONE', 'JUPITER')
    
    x = positions.T[0]/69911
    y = positions.T[1]/69911
    z = positions.T[2]/69911
    
    plt.plot(x, y)
        
    plt.grid()
    plt.show()
    
    lat_df = time_in_lat_window(start, end)
    
    for index, window in lat_df.iterrows():
        time_series = pd.date_range(window['START'].isoformat(), window['END'].isoformat(), freq='1H')
        et_range = [spice.utc2et(i) for i in time_series.strftime('%Y-%m-%dT%H:%M:%S')]

        positions, lt = spice.spkpos('JUNO', et_range, 'JUNO_JSS', 'NONE', 'JUPITER')
        
        x = positions.T[0]/69911
        y = positions.T[1]/69911
        z = positions.T[2]/69911
        
        plt.plot(x, y)
        
    plt.grid()
    plt.show()
    spice.kclear()
    
    
def radial_q_kaw_for_poster():
    q_plot = QKAWData('/home/aschok/Documents/data/heating_data_+-10')
    
    fig, ax = plt.subplots(figsize=(15,7))
    q_v_pos_data = q_plot.q_v_pos
    bins = np.arange(np.floor(np.min(q_v_pos_data.radial)) - 0.5,
                    np.ceil(np.max(q_v_pos_data.radial)) + 1.5)
    means, edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                        'mean', bins)
    bin_std, std_edges, num = binned_statistic(q_v_pos_data.radial, q_v_pos_data.q_kaw,
                                            'std', bins)
    edges += 0.5
    ax.plot(edges[:-1], means, label='avg $q_{kaw}$')
    ax.plot(edges[:-1], means + bin_std, label='avg $q_{kaw}$ + 1$\sigma$')
    ax.set_xlabel('Radial Distance (Rj)')
    ax.set_ylabel('$q_{kaw}$')
    ax.set_title(f'$q_{{kaw}}$ vs. Radial Distance')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'/home/aschok/Documents/figures/q_kaw/q_kaw_radial.png')
    plt.close(fig)  
    
if __name__ == '__main__':
    orbits_plot()
