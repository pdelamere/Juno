from juno_classes import MagData, CWTData, Turbulence
from juno_functions import time_in_lat_window
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def save():
    start = '2016-07-31T00:00:00'
    end = '2017-01-07T00:00:00'

    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        mag_class = MagData(row['START'].isoformat(), row['END'].isoformat())
        mag_class.downsample_data(60)
        mag_class.mean_field_align(60)
        time_series = mag_class.data_df.index
        b_perp = np.sqrt(mag_class.data_df.B_PERP2.to_numpy()**2
                        + mag_class.data_df.B_PERP1.to_numpy()**2)
        cwt = CWTData(time_series, b_perp, 60)
        file_name = f'{row["START"].date()}_{row["END"].date()}_cwt.pickle'
        file_loc = f'/home/aschok/Documents/data/{file_name}'
        with open(file_loc, 'wb') as file:
            
            pickle.dump({'notes': 'mag data downsampled to 60 second inervals and MFA using a 60 minute window',
                        'time': cwt.time_series,
                        'freqs': cwt.freqs,
                        'coi': cwt.coi,
                        'power': cwt.power}, file, protocol=4)
            file.close()
        print('Saved data')
        
def save_heating():
    start = '2016-07-31T00:00:00'
    end = '2020-11-10T00:00:00'
    
    lat_df = time_in_lat_window(start, end)
    for index, row in lat_df.iterrows():
        start_datetime = row['START']
        end_datetime = row['END']
        file = f'q_data_{start_datetime.date()}-{end_datetime.date()}.pickle'
    
        turb = Turbulence(start_datetime.isoformat(),
                          end_datetime.isoformat(),
                          1, 60, 30)
        file_path = f'/home/aschok/Documents/data/heating_data/{file}'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(turb.q_data, pickle_file)
            print(f'Saved data from {start_datetime} to {end_datetime}')
            pickle_file.close()
    
if __name__ == '__main__':
    save()