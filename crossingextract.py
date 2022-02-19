import csv, datetime, pathlib, os, re, pickle
import pandas as pd
import numpy as np
from spacepy import pycdf
import spiceypy as spice
from old.spacedataclasses import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates

class CrossingClass():
    
    def __init__(self,crossing_time,crossing_type,width):
        self.data = pd.DataFrame({'TIME':[], 'BX DATA':[], 'BY DATA':[], 'BZ DATA':[],
                                  'X POSITION':[], 'Y POSITION':[], 'Z POSITION':[],
                                  'RADIAL DISTANCE':[], 'LATITUDE':[]})
        self.crossing_time = datetime.datetime.fromisoformat(crossing_time)
        self.width = int(width)
        self.type = crossing_type
        self.date_range = [self.crossing_time-datetime.timedelta(hours=self.width),self.crossing_time+datetime.timedelta(hours=self.width)]
        self.jade_data,self.jade_time,self.jade_dim1 = [],[],[]
        
        
    def getMag(self,fgm_folder):
                
        temp_df = pd.DataFrame()
        
        p = re.compile(r'\d{7}')
        
        for parent,child,files in os.walk(fgm_folder):
            for file_name in files:
                if file_name.endswith('.csv'):
                    file_path = os.path.join(fgm_folder,file_name)
                    fgm_date = p.search(file_name).group()
                    fgm_date = datetime.datetime.strptime(fgm_date,'%Y%j')
                    
                    if self.date_range[0].date() == fgm_date.date() or self.date_range[1].date() == fgm_date.date():
                        
                        fgm_data = pd.read_csv(file_path)
                        temp_df = temp_df.append(fgm_data,ignore_index=True)
        temp_df = temp_df.sort_values(by=['SAMPLE UTC'])    
        self.time_list = [datetime.datetime.fromisoformat(i) for i in temp_df['SAMPLE UTC']]
        crossing_index_begin = int(self.time_list.index(min(self.time_list,key=lambda x: abs(x-(self.date_range[0])))))
        crossing_index_end = int(self.time_list.index(min(self.time_list,key=lambda x: abs(x-(self.date_range[1])))))
        self.data['TIME'] = temp_df['SAMPLE UTC'][crossing_index_begin:crossing_index_end+1]
        self.data['BX DATA'] = temp_df['BX PLANETOCENTRIC'][crossing_index_begin:crossing_index_end+1]
        self.data['BY DATA'] = temp_df['BY PLANETOCENTRIC'][crossing_index_begin:crossing_index_end+1]
        self.data['BZ DATA'] = temp_df['BZ PLANETOCENTRIC'][crossing_index_begin:crossing_index_end+1]
        print(f'Mag data pulled {self.crossing_time}')
        
    def getJade(self,jade_folder):
        
        DOY,ISO,datFiles = getFiles(self.date_range[0].isoformat(),self.date_range[1].isoformat(),'.DAT',jade_folder,'JAD_L30_LRS_ION_ANY_CNT') 
        if len(datFiles) == 0:
            self.jade_data = np.array([])
            self.jade_time = np.array([])
            self.jade_dim1 = np.array([])
           
        else:    
            jadeIon = JadeData(datFiles,self.date_range[0].isoformat(),self.date_range[1].isoformat())
            jadeIon.getIonData()
            
            temp = []
            jade_data = []
            jade_time = []
            jade_dim1 = []
            for date in jadeIon.dataDict.keys():
                temp = np.append(temp,jadeIon.dataDict[date]['DATETIME_ARRAY'])
                jade_data = np.append(jade_data,jadeIon.dataDict[date]['DATA_ARRAY'])
                jade_dim1 = jadeIon.dataDict[date]['DIM1_ARRAY']
            jade_time_list = [datetime.datetime.fromisoformat(i) for i in temp]
            crossing_index_begin = int(jade_time_list.index(min(jade_time_list,key=lambda x: abs(x-(self.date_range[0])))))
            crossing_index_end = int(jade_time_list.index(min(jade_time_list,key=lambda x: abs(x-(self.date_range[1])))))
            matrix = np.transpose(np.reshape(jade_data,(int(len(jade_data)/64),64)))
            self.jade_data = np.flip(matrix[:,crossing_index_begin:crossing_index_end+1],axis=0)
            self.jade_time = temp[crossing_index_begin:crossing_index_end+1]
            self.jade_dim1 = jade_dim1
        
        print(f'Ion data pulled {self.crossing_time}')
        DOY,ISO,datFiles = getFiles(self.date_range[0].isoformat(),self.date_range[1].isoformat(),'.DAT',jade_folder,'JAD_L30_LRS_ELC_ANY_CNT') 
        if len(datFiles) == 0:
            self. jade_elec_data = np.array([])
            self.jade_elec_time = np.array([])
            self.jade_elec_dim1 = np.array([])
        else:    
            jadeElec = JadeData(datFiles,self.date_range[0].isoformat(),self.date_range[1].isoformat())
            jadeElec.getElecData()
            
            temp = []
            jade_elec_data = []
            jade_elec_time = []
            jade_elec_dim1 = []
            for date in jadeElec.dataDict.keys():
                temp = np.append(temp,jadeElec.dataDict[date]['DATETIME_ARRAY'])
                jade_elec_data = np.append(jade_elec_data,jadeElec.dataDict[date]['DATA_ARRAY'])
                jade_elec_dim1 = jadeElec.dataDict[date]['DIM1_ARRAY']
            jade_elec_time_list = [datetime.datetime.fromisoformat(i) for i in temp]
            crossing_index_begin = int(jade_elec_time_list.index(min(jade_elec_time_list,key=lambda x: abs(x-(self.date_range[0])))))
            crossing_index_end = int(jade_elec_time_list.index(min(jade_elec_time_list,key=lambda x: abs(x-(self.date_range[1])))))
            matrix = np.transpose(np.reshape(jade_elec_data,(int(len(jade_elec_data)/64),64)))
            self.jade_elec_data = matrix[:,crossing_index_begin:crossing_index_end+1]
            self.jade_elec_time = temp[crossing_index_begin:crossing_index_end+1]
            self.jade_elec_dim1 = jade_elec_dim1
        
        
    def getPosition(self,meta_kernel):

        spice.furnsh(meta_kernel)
        spice_time_list = [spice.utc2et(i) for i in self.data['TIME']]
        position_list = []
        latitude_list = []
        x,y,z=[],[],[]
        for spice_time in spice_time_list:
            pos,lt = spice.spkpos('JUNO',spice_time,'JUNO_JSS','NONE','JUPITER')
            pos_vec = spice.vpack(pos[0],pos[1],pos[2])
            rad_pos,long,lat = spice.reclat(pos_vec)
            lat *= spice.dpr()
            rad_pos /= 69911

            pos,lt = spice.spkpos('JUNO',spice_time,'JUNO_JSS','NONE','JUPITER')
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])

            position_list.append(rad_pos)
            latitude_list.append(lat)
        spice.kclear()
        self.data['RADIAL DISTANCE'] = position_list
        self.data['LATITUDE'] = latitude_list
        self.data['X POSITION'] = x
        self.data['Y POSITION'] = y
        self.data['Z POSITION'] = z

        print(f'Position data pulled {self.crossing_time}')

    def packageCDF(self,save_location):
        file_save_date = self.crossing_time.strftime('%Y%jT%H%M%S') + f'_{self.type}'
        save_name = str(pathlib.Path(f'{save_location}/{file_save_date}'))
        cdf_file = pycdf.CDF(save_name,'')

        cdf_file.attrs['Author'] = 'Andrew Schok'
        cdf_file['RADIAL DISTANCE'] = self.data['RADIAL DISTANCE']
        cdf_file['MAG TIME'] = self.data['TIME'].tolist()
        cdf_file['BX DATA'] = self.data['BX DATA'].tolist()
        cdf_file['BX DATA'].attrs['units'] = 'nT'
        cdf_file['BY DATA'] = self.data['BY DATA'].tolist()
        cdf_file['BY DATA'].attrs['units'] = 'nT'
        cdf_file['BZ DATA'] = self.data['BZ DATA'].tolist()
        cdf_file['BZ DATA'].attrs['units'] = 'nT'
        cdf_file['X POSITION'] = self.data['X POSITION'].tolist()
        cdf_file['X POSITION'].attrs['units'] = 'km'
        cdf_file['Y POSITION'] = self.data['Y POSITION'].tolist()
        cdf_file['Y POSITION'].attrs['units'] = 'km'
        cdf_file['Z POSITION'] = self.data['Z POSITION'].tolist()
        cdf_file['Z POSITION'].attrs['units'] = 'km'
        cdf_file['RADIAL DISTANCE'] = self.data['RADIAL DISTANCE'].tolist()
        cdf_file['RADIAL DISTANCE'].attrs['units'] = 'Rj'
        cdf_file['LATITUDE'] = self.data['LATITUDE'].tolist()
        cdf_file['LATITUDE'].attrs['units'] = 'deg'
        cdf_file['JADE ION TIME'] = self.jade_time
        cdf_file['JADE ION TIME'].attrs['units'] = 'seconds'
        cdf_file['JADE ION DATA'] = self.jade_data
        cdf_file['JADE ION DATA'].attrs['units'] = 'log(counts/sec)'
        cdf_file['JADE ION Y-AXIS'] = self.jade_dim1
        cdf_file['JADE ION Y-AXIS'].attrs['units'] = 'KeV/q'
        cdf_file['JADE ELEC TIME'] = self.jade_elec_time
        cdf_file['JADE ELEC TIME'].attrs['units'] = 'seconds'
        cdf_file['JADE ELEC DATA'] = self.jade_elec_data
        cdf_file['JADE ELEC DATA'].attrs['units'] = 'log(counts/sec)'
        cdf_file['JADE ELEC Y-AXIS'] = self.jade_elec_dim1
        cdf_file['JADE ELEC Y-AXIS'].attrs['units'] = 'KeV/q'       
        cdf_file.close()
        print(f'Created CDF for {self.type} crossing {self.crossing_time.strftime("%Y-%m-%dT%H:%M:%S")}\n')                     
                        
def textract():
    crossing_file = r'/data/juno_spacecraft/data/crossings/crossingmasterlist/jno_crossings_master_v6.txt'
    fgm_folder = r"/data/juno_spacecraft/data/fgm"
    jade_folder = r"/data/juno_spacecraft/data/jad"
    save_loc = r'/home/delamere/Documents/cdf'
    
    crossing_list = pd.read_csv(crossing_file, skiprows=[1])
    
    for index, row in crossing_list.iterrows():
        crossing = f'{row["DATE"]}T{row["TIME"]}'
        crossing_type = row['BOUNDARYID']
        crossing_datetime = datetime.datetime.fromisoformat(crossing)
        meta_kernel = (f'/data/juno_spacecraft/data/meta_kernels/juno_{crossing_datetime.year}.tm')
        
        crossings = CrossingClass(crossing,crossing_type,2)
        crossings.getMag(fgm_folder)
        try: crossings.getJade(jade_folder)
        except ValueError: pass
        crossings.getPosition(meta_kernel)       
        crossings.packageCDF(save_loc)            

def single_extract():
    crossings = ['2017-05-06T20:30:24', '2017-06-02T15:08:00', 
                 '2017-06-16T22:15:00', '2017-06-28T21:42:18.500000',
                 '2017-10-02T23:16:11.500000']
    cross_type = ['Long']
    
    fgm_folder = r"/data/juno_spacecraft/data/fgm"
    jade_folder = r"/data/juno_spacecraft/data/jad"
    save_loc = r'/home/delamere/Documents'
    meta_kernel = (f'/data/juno_spacecraft/data/meta_kernels/juno_2017.tm')
    for i, time in enumerate(crossings):
        crossings = CrossingClass(time, cross_type[i], 2)
        crossings.getMag(fgm_folder)
        try: crossings.getJade(jade_folder)
        except ValueError: pass
        crossings.getPosition(meta_kernel)       
        crossings.packageCDF(save_loc)

def custom_intervals():
    intervals_df=pd.DataFrame({'Start': ['2017-05-07T00:00:24',
					 '2017-06-02T16:38:00'],
                               'End': ['2017-05-07T04:30:00',
				       '2017-06-02T21:00:00']})
    fgm_folder = r"/data/juno_spacecraft/data/fgm"
    jade_folder = r"/data/juno_spacecraft/data/jad"
    save_loc = r'/home/delamere/Documents/cdf'
    meta_kernel = (f'/data/juno_spacecraft/data/meta_kernels/juno_2017.tm')
    for index, row in intervals_df.iterrows():
        start = datetime.datetime.fromisoformat(row.Start)
        end = datetime.datetime.fromisoformat(row.End)
        cross_time = (start + (end - start)/2).isoformat()
        cross_width = ((end-start)/2).total_seconds()/3600
        crossings = CrossingClass(cross_time, 'long', cross_width)
        crossings.getMag(fgm_folder)
        try: crossings.getJade(jade_folder)
        except ValueError: pass
        crossings.getPosition(meta_kernel)       
        crossings.packageCDF(save_loc)

def test():
    
    fgm_folder = r"/data/Python/jupiter/data/fgm"
    jade_folder = r"/data/Python/jupiter/data/jad"
    meta_kernel = r'juno_2017.tm'
    save_loc = r'/home/delamere/Documents'
        
    cdf = pycdf.CDF('/home/delamere/Documents/2017042T030000_Sheath.cdf')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    time = [datetime.datetime.fromisoformat(i) for i in cdf['MAG TIME']]
    ax1.plot(time, cdf['BX DATA'])
    ax1.plot(time, cdf['BY DATA'])
    ax1.plot(time, cdf['BZ DATA'])
    
    time = [datetime.datetime.fromisoformat(i) for i in cdf['JADE ION TIME']]
    t = mdates.date2num(time)
    
    ax2.pcolormesh(t, np.array(cdf['JADE ION Y-AXIS'])/1000, cdf['JADE ION DATA'], cmap='jet')
    ax2.set_yscale('log')
    
    time = [datetime.datetime.fromisoformat(i) for i in cdf['JADE ELEC TIME']]
    t = mdates.date2num(time)
    
    ax3.pcolormesh(t, np.array(cdf['JADE ELEC Y-AXIS'])/1000, cdf['JADE ELEC DATA'], cmap='jet')
    ax3.set_yscale('log')
    plt.show()
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def dataToPickle():
    orbits_begin = {1:'2016-07-31T19:46:02',
                            2:'2016-09-23T03:44:48',
                            3:'2016-11-15T05:36:45',
                            4:'2017-01-07T03:11:30',
                            5:'2017-02-28T22:55:48',
                            6:'2017-04-22T19:14:57'}
    
    file_dict = {}
    metaKernel = 'juno_2019_v03.tm'
    spice.furnsh(metaKernel)

    start_time = datetime.datetime.strptime(orbits_begin[1],'%Y-%m-%dT%H:%M:%S')
    
    end_time = datetime.datetime.strptime(orbits_begin[2],'%Y-%m-%dT%H:%M:%S')
    
    data_folder = pathlib.Path(r'..\data\fgm')
    p = re.compile(r'\d{7}')
    
    for parent,child,files in os.walk(data_folder):
        for name in files:
            if name.endswith('.csv'):
                file_path = os.path.join(data_folder,name)
                
                search = p.search(name).group()
                date = datetime.datetime.strptime(search,'%Y%j')
                
                if date.date() >= start_time.date() and date.date() <= end_time.date():
                    iso_date = date.strftime('%Y-%m-%d')
                    if iso_date not in file_dict.keys():
                        file_dict[iso_date] = [file_path]
                    elif iso_date in file_dict.keys() and file_dict[iso_date] != file_path: 
                        file_dict[iso_date].append(file_path)
    
    for date in file_dict.keys():
        fgmdf = pd.DataFrame(data={'TIME':[],'BX':[],'BY':[],'BZ':[],'LAT':[]})
        save_date = datetime.datetime.strptime(date,'%Y-%m-%d')
        file_list = file_dict[date]
        for file in file_list:
            
            temp = pd.read_csv(file)
            datetime_list = temp['SAMPLE UTC']
            time_list = [datetime.datetime.fromisoformat(i).strftime('%H:%M:%S') for i in datetime_list]
            
            for index,time in enumerate(datetime_list):
                
                position, lighttime = spice.spkpos('JUNO',spice.utc2et(time),'IAU_JUPITER','NONE','JUPITER')
            
                vectorPos = spice.vpack(position[0],position[1],position[2])
                radii,longitude,latitude = spice.reclat(vectorPos)
                lat = latitude*spice.dpr()
                
                if lat >= -10 and lat <= 10:
                    fgmdf = fgmdf.append({'TIME':time,'BX':temp['BX PLANETOCENTRIC'][index],'BY':temp['BY PLANETOCENTRIC'][index],'BZ':temp['BZ PLANETOCENTRIC'][index],'LAT':lat},ignore_index=True)
        fgmdf = fgmdf.sort_values(by=['TIME'])
        save_name = f'{save_date.strftime("%Y%m%d")}'
        save_path = pathlib.Path(f'..\data\pickledfgm\jno_fgm_{save_name}.pkl')
        pickledf = fgmdf.to_pickle(save_path)
        print(f'Saved pickle {date}')                                     
#----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    custom_intervals()
    
    
    
    
