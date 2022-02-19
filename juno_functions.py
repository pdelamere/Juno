import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import spiceypy as spice


def time_in_lat_window(start_time, end_time, max_lat=10, min_lat=-10):
    """

    Parameters
    ----------
    start_time : string
        Date sting in iso format. '2016-01-01T00:00:00'
    end_time : string
        Date sting in iso format. '2016-01-01T00:00:00'
    target_latitude : int, optional
        latitude which you want to find the date range in which juno is +- latitude.
        Default is 10.

    Returns
    -------
    lat_windows_df : pandas Dataframe
        Dataframe with time the spacecraft enters and leaves the interval of +- latitude.
            Columns: 'START', 'END', 'START_LAT', 'END_LAT'.

    """
    start_time = datetime.fromisoformat(start_time)

    end_time = datetime.fromisoformat(end_time)

    time_list = pd.date_range(start_time, end_time, freq='min')
    for load_date in time_list.year.unique():
        meta_kernel = f'juno_{load_date}.tm'
        spice.furnsh(f'/data/juno_spacecraft/data/meta_kernels/{meta_kernel}')
    lat_windows_df = pd.DataFrame({'START': [],
                                   'END': [],
                                   'START_LAT': [],
                                   'END_LAT': []})

    # Finds the dates in the given time frame when juno enters and leave +-10 degrees latitude.
    iso_time_stamp = time_list[0].isoformat()
    position, lighttime = spice.spkpos('JUNO', spice.utc2et(iso_time_stamp),
                                       'JUNO_JSS', 'NONE', 'JUPITER')
    vector_pos = spice.vpack(position[0], position[1], position[2])
    radii, longitude, latitude = spice.reclat(vector_pos)
    lat_ini = latitude*spice.dpr()
    time_stamp_ini = time_list[0]
    if min_lat <= lat_ini <= max_lat:
        enter_time = time_stamp_ini
        enter_lat = lat_ini
        entered = True
        exited = False
    else:
        entered = False
        exited = False
    for i, time_stamp in enumerate(time_list[1:]):
        iso_time_stamp = time_stamp.isoformat()
        position, lighttime = spice.spkpos('JUNO', spice.utc2et(iso_time_stamp),
                                           'JUNO_JSS', 'NONE', 'JUPITER')
        vector_pos = spice.vpack(position[0], position[1], position[2])
        radii, longitude, latitude = spice.reclat(vector_pos)
        lat = latitude*spice.dpr()
        if min_lat <= lat <= max_lat:
            if np.abs(lat - lat_ini) > 1:
                pass
            elif lat_ini < min_lat or lat_ini > max_lat:
                enter_time = time_stamp
                enter_lat = lat
                entered = True
        if min_lat <= lat_ini <= max_lat:
            if np.abs(lat - lat_ini) > 1:
                pass
            elif lat < min_lat or lat > max_lat or i + 2 == len(time_list):
                exit_time = time_stamp_ini
                exit_lat = lat_ini
                exited = True
        if entered and exited:
            lat_windows_df = lat_windows_df.append({'START': enter_time, 'END': exit_time,
                                                    'START_LAT': enter_lat, 'END_LAT': exit_lat},
                                                   ignore_index=True)
            entered = False
            exited = False
        time_stamp_ini = time_stamp
        lat_ini = lat
    spice.kclear()
    return lat_windows_df

def get_sheath_intervals(crossing_file):
    crossings_df = pd.read_csv(crossing_file)
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
    return sheath_windows_df

def _get_files(start_time, end_time, file_type, data_folder, *args):
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

    datetime_array = pd.date_range(start_time, end_time, freq='D').date
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

def find_orb_num(date_time):
    if type(date_time) is str:
        date_time = datetime.fromisoformat(date_time)
        
    orbs = pd.read_fwf('/data/juno_spacecraft/data/orbits/juno_rec_orbit_v08.orb')
    orbs = orbs.drop(index=[0])
    for index in orbs.index[:-1]:
        orb_start = datetime.strptime(orbs['Event UTC APO'][index], '%Y %b %d %H:%M:%S')
        orb_end = datetime.strptime(orbs['Event UTC APO'][index + 1], '%Y %b %d %H:%M:%S')
        if (date_time > orb_start) & (date_time < orb_end):
            return orbs['No.'][index]
        