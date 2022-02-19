from juno_classes import JadeData, MagData, CWTData
import pandas as pd
from juno_functions import _get_files, get_sheath_intervals, find_orb_num
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import spiceypy as spice
import re
import os
import pickle
import scipy

class QKAWData:
    def __init__(self, data_folder, minlat=-10, maxlat=0):
        self.data_folder = data_folder
        self.minlat = minlat
        self.maxlat = maxlat
        self.q_kaw_df = self._get_data()
        
    def _get_data(self):
        # Data files are opened and the data is read in and cleaned
        temp = pd.DataFrame()
        r = re.compile('_data_')
        for root, dirs, files in os.walk(self.data_folder):
            for file_name in files:
                if r.search(file_name):
                    with open(os.path.join(root, file_name), 'rb') as data_file:
                        pkl = pickle.load(data_file)
                        temp = pd.concat([temp, pkl])
                        temp = temp[np.isfinite(temp.q_kaw)]
        return temp
    
    def remove_sheath(self):
        sheath_df = get_sheath_intervals('/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_v6.txt')
        sheath_window = False
        data = self.q_kaw_df
        temp_data = pd.DataFrame()
        for index, row in sheath_df.iterrows():
            if (self.q_kaw_df.index.min() < row.START < self.q_kaw_df.index.max()) or\
            (self.q_kaw_df.index.min() < row.END < self.q_kaw_df.index.max()):
                sheath_window = True
                data = data[(data.index < row.START) | (data.index > row.END)]
        if sheath_window:
            self.q_kaw_df = data
        
    def sys_3_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
            
        index_array = self.q_kaw_df.index
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
        z_equator = positions.T[2]/7.14e4 - R*np.sin(CentEq2)
        temp_df = pd.DataFrame({'radial_3': rad/7.14e4, 'lon_3': lon,
                                'lat_3': lat, 'eq_dist': z_equator}, index=index_array)
        
        self.q_kaw_df = pd.concat([self.q_kaw_df.sort_index(), temp_df.sort_index()], axis=1)

    def despun_sys_data(self):
        for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
            
        index_array = self.q_kaw_df.index
        et_array = [spice.utc2et(i) for i in index_array.strftime('%Y-%m-%dT%H:%M:%S')]
        positions, lt = spice.spkpos('JUNO', et_array,
                                    'JUNO_JSS', 'NONE', 'JUPITER')
        
        x = np.array(positions.T[0])
        y = np.array(positions.T[1])
        z = np.array(positions.T[2])
        rad = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z/rad)*180/np.pi
        long = np.arctan2(y,x)*180/np.pi
        spice.kclear()
        
        temp_df = pd.DataFrame({'radial': rad/7.14e4, 'longitude': long,
                                'latitude': lat}, index=index_array)
        
        self.q_kaw_df = pd.concat([self.q_kaw_df.sort_index(), temp_df.sort_index()], axis=1)

        
def kaw():
    q_kaw = QKAWData('/home/aschok/Documents/data/heating_data_+-10')
    q_kaw.remove_sheath()
    q_kaw.sys_3_data()
    data = q_kaw.q_kaw_df
    hist_data = data
    data_2 = data[(data.eq_dist > -20) & (data.eq_dist < 20)\
                    & (data.radial_3 > 80)]
    data_3 = data[(data.eq_dist > -20) & (data.eq_dist < 20)\
                    & (data.radial_3 > 60) & (data.radial_3 < 80)]
    data_4 = data[(data.eq_dist > -20) & (data.eq_dist < 20)\
                    & (data.radial_3 < 60)]
    data = data[(data.eq_dist > -2) & (data.eq_dist < 2)]

    def get_25perc(arr):
        return np.percentile(arr, 25)
    def get_75perc(arr):
        return np.percentile(arr, 75)

    bin_num = 100
    median, edges, num = scipy.stats.binned_statistic(data.radial_3, data.q_kaw, 'median', bin_num)
    mean, edges, num = scipy.stats.binned_statistic(data.radial_3, data.q_kaw, 'mean', bin_num)
    std, edges, num = scipy.stats.binned_statistic(data.radial_3, data.q_kaw, 'std', bin_num)
    perc_25, edges, num = scipy.stats.binned_statistic(data.radial_3, data.q_kaw, get_25perc, bin_num)
    perc_75, edges, num = scipy.stats.binned_statistic(data.radial_3, data.q_kaw, get_75perc, bin_num)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True)
    ax1.scatter(data.radial_3, data.q_kaw, 3)
    ax1.plot(edges[:-1], median, color='black')
    ax1.plot(edges[:-1], perc_25, color='red')
    ax1.plot(edges[:-1], perc_75, color='red')
    ax1.set_yscale('log')
    ax1.set_xlabel('Radial Distance (Rj)')
    ax1.set_ylabel('$q_{KAW}$')

    bin_num = 100
    median, edges, num = scipy.stats.binned_statistic(data_2.eq_dist, data_2.q_kaw, 'median', bin_num)
    mean, edges, num = scipy.stats.binned_statistic(data_2.eq_dist, data_2.q_kaw, 'mean', bin_num)
    std, edges, num = scipy.stats.binned_statistic(data_2.eq_dist, data_2.q_kaw, 'std', bin_num)
    perc_25, edges, num = scipy.stats.binned_statistic(data_2.eq_dist, data_2.q_kaw, get_25perc, bin_num)
    perc_75, edges, num = scipy.stats.binned_statistic(data_2.eq_dist, data_2.q_kaw, get_75perc, bin_num)

    ax2.scatter(data_2.eq_dist, data_2.q_kaw, 3)
    ax2.set_title('Radial Distance > 80 Rj')
    ax2.plot(edges[:-1], median, color='black')
    ax2.plot(edges[:-1], perc_25, color='red')
    ax2.plot(edges[:-1], perc_75, color='red')
    ax2.set_yscale('log')
    ax2.set_xlabel('Cenrifugal Equatorial Dist. (Rj)')
    ax2.set_ylabel('$q_{KAW}$')
    ax2.set_xlim(-20, 20)

    bin_num = 100
    median, edges, num = scipy.stats.binned_statistic(data_3.eq_dist, data_3.q_kaw, 'median', bin_num)
    mean, edges, num = scipy.stats.binned_statistic(data_3.eq_dist, data_3.q_kaw, 'mean', bin_num)
    std, edges, num = scipy.stats.binned_statistic(data_3.eq_dist, data_3.q_kaw, 'std', bin_num)
    perc_25, edges, num = scipy.stats.binned_statistic(data_3.eq_dist, data_3.q_kaw, get_25perc, bin_num)
    perc_75, edges, num = scipy.stats.binned_statistic(data_3.eq_dist, data_3.q_kaw, get_75perc, bin_num)


    ax3.scatter(data_3.eq_dist, data_3.q_kaw, 3)
    ax3.set_title('60 Rj < Radial Distance < 80 Rj')
    ax3.plot(edges[:-1], median, color='black')
    ax3.plot(edges[:-1], perc_25, color='red')
    ax3.plot(edges[:-1], perc_75, color='red')
    ax3.set_yscale('log')
    ax3.set_xlabel('Cenrifugal Equatorial Dist. (Rj)')
    ax3.set_ylabel('$q_{KAW}$')
    ax3.set_xlim(-20, 20)

    bin_num = 100
    median, edges, num = scipy.stats.binned_statistic(data_4.eq_dist, data_4.q_kaw, 'median', bin_num)
    mean, edges, num = scipy.stats.binned_statistic(data_4.eq_dist, data_4.q_kaw, 'mean', bin_num)
    std, edges, num = scipy.stats.binned_statistic(data_4.eq_dist, data_4.q_kaw, 'std', bin_num)
    perc_25, edges, num = scipy.stats.binned_statistic(data_4.eq_dist, data_4.q_kaw, get_25perc, bin_num)
    perc_75, edges, num = scipy.stats.binned_statistic(data_4.eq_dist, data_4.q_kaw, get_75perc, bin_num)


    ax4.scatter(data_4.eq_dist, data_4.q_kaw, 3)
    ax4.set_title('Radial Distance < 60 Rj')
    ax4.plot(edges[:-1], median, color='black')
    ax4.plot(edges[:-1], perc_25, color='red')
    ax4.plot(edges[:-1], perc_75, color='red')
    ax4.set_yscale('log')
    ax4.set_xlabel('Cenrifugal Equatorial Dist. (Rj)')
    ax4.set_ylabel('$q_{KAW}$')
    ax4.set_xlim(-20, 20)

    fig.subplots_adjust(wspace=0.1, hspace=0.7)
    plt.show()

    bins = np.logspace(-19, -13, num=25)

    hist1_data = hist_data[(hist_data.eq_dist > 10) | (hist_data.eq_dist < -10)\
                        & (hist_data.radial_3 < 60)]
    hist2_data = hist_data[(hist_data.eq_dist < 2) & (hist_data.eq_dist > -2)\
                        & (hist_data.radial_3 < 60)]
    fig, ax = plt.subplots()
    ax.hist(np.array([hist1_data.q_kaw, hist2_data.q_kaw]), bins=bins, log=True)
    # ax.hist(hist2_data.q_kaw, bins=bins, log=True)
    ax.set_xlabel('q_kaw')
    ax.set_xscale('log')
    ax.set_title('< 60 Rj')

    hist3_data = hist_data[(hist_data.eq_dist > 10) | (hist_data.eq_dist < -10)\
                        & (hist_data.radial_3 < 80) & (hist_data.radial_3 > 60)]
    hist4_data = hist_data[(hist_data.eq_dist < 2) & (hist_data.eq_dist > -2)\
                        & (hist_data.radial_3 < 80) & (hist_data.radial_3 > 60)]
    fig, ax = plt.subplots()
    ax.hist(np.array([hist3_data.q_kaw, hist4_data.q_kaw]), bins=bins, log=True)
    ax.set_xlabel('q_kaw')
    ax.set_xscale('log')
    ax.set_title('60 Rj < R < 80 Rj')

    hist5_data = hist_data[(hist_data.eq_dist > 10) | (hist_data.eq_dist < -10)\
                        & (hist_data.radial_3 > 80)]
    hist6_data = hist_data[(hist_data.eq_dist < 2) & (hist_data.eq_dist > -2)\
                        & (hist_data.radial_3 > 80)]
    fig, ax = plt.subplots()
    ax.hist(np.array([hist5_data.q_kaw, hist6_data.q_kaw]), bins=bins, log=True)
    ax.set_xlabel('q_kaw')
    ax.set_xscale('log')
    ax.set_title('> 80 Rj')

    plt.show()

def sys3_mag():
    mag = MagData('2017-05-15T06:00:00', '2017-05-18T22:30',instrument=['fgm_jno', 'r60s'])
    mag_data = mag.data_df
    for year in ['2016', '2017', '2018', '2019', '2020']:
                    spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')
                    
    et_array = [spice.utc2et(i) for i in mag_data.index.strftime('%Y-%m-%dT%H:%M:%S')]
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

    sin_long = np.sin((lon - 292)*np.pi/180)*np.sqrt(x**2 + y**2)
    z_equator = positions.T[2]/7.14e4 - R*np.sin(CentEq2)

    mask = (z_equator < 2) & (z_equator > -2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.plot(mag_data.index, mag_data.BX)
    ax1.plot(mag_data.index, mag_data.BY)
    ax1.plot(mag_data.index, mag_data.BZ)
    ax1.plot(mag_data.index, np.sqrt(mag_data.BX**2 + mag_data.BY**2 + mag_data.BZ**2), 'black')
    ax2.plot(mag_data.index, z_equator)
    ax2.axhline(0, linestyle='--', color='black')
    ax2.set_ylim(-4, 4)
    plt.show()

    # q_data = QKAWData('/home/aschok/Documents/data/heating_data_+-10')
    # q_v_pos_data = q_data.q_v_pos
    # filtered_data = q_v_pos_data[(q_v_pos_data.z_eq > -2) & (q_v_pos_data < 2)]

    # ax3.plot(filtered_data.radial, filtered_data.q_kaw_array)

    # plt.show()

def wiggle_plots():
    from scipy.spatial.transform import Rotation as R
    for year in ['2016', '2017', '2018', '2019', '2020']:
                spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/juno_{year}.tm')

    date_range = pd.date_range('2017-06-14', '2017-08-16', freq='30min')
    index_array = date_range
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

    x_3 = np.array(positions.T[0])
    y_3 = np.array(positions.T[1])
    z_3 = np.array(positions.T[2])
    

    deg2rad = np.pi/180
    a = 1.66*deg2rad
    b = 0.131
    R = np.sqrt(x_3**2 + y_3**2 + z_3**2)/7.14e4
    c = 1.62
    d = 7.76*deg2rad
    e = 249*deg2rad
    CentEq2 = (a*np.tanh(b*R - c) + d)*np.sin(lon*deg2rad - e)
    z_equator = z_3/7.14e4 - R*np.sin(CentEq2)
    print(z_3/7.14e4+z_equator)
    
    positions, lt = spice.spkpos('JUNO', et_array,
                                'JUNO_JSS', 'NONE', 'JUPITER')
    spice.kclear()
    fig, ax = plt.subplots()
    ax.plot(positions.T[0]/7.14e4, positions.T[2]/7.14e4 + z_equator)
    plt.show()
    
def cwt_analysis():
    from scipy.signal import find_peaks
    start = '2017-01-07T00:00:00'
    end = '2017-02-28T00:00:00'
    
    start_dt = datetime.fromisoformat(start)
    intervals = np.ceil((datetime.fromisoformat(end)-start_dt)/timedelta(days=2))
    psd_hist = np.array([])
    for _ in range(int(intervals)):
        next_dt = start_dt + timedelta(days=2)
        mag = MagData(start_dt.isoformat(), next_dt.isoformat())
        mag.downsample_data(60)
        mag.mean_field_align(30)
        b_perp = np.sqrt(mag.data_df.B_PERP1**2 + mag.data_df.B_PERP2**2).to_numpy()
        cwt = CWTData(mag.data_df.index, b_perp, 60)
        fig, (ax1, ax2) = plt.subplots(2)
        cwt.cwt_plot(ax1)
        cwt.psd_plot(ax2, 'sec')
        ax2.set_xscale('log')
        peaks, _ = find_peaks(cwt.psd, width=1)
        ax2.scatter(cwt.freqs[peaks], cwt.psd[peaks])
        psd_hist = np.append(psd_hist, cwt.freqs[peaks])
        plt.savefig(f'/home/aschok/Documents/figures/cwt/psd_periods_cnt/{start_dt}')
        plt.close(fig)
        start_dt = next_dt
    fig, ax = plt.subplots()
    ax.hist(psd_hist)
    ax.set_xscale('log')
    plt.show()
    
if __name__ == '__main__':
    cwt_analysis()