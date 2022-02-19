#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:09:30 2021

@author: delamere
"""
import pathlib,datetime,logging
#-------------------------------------------------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------------------------------------------------
class FGMData():
    """Class for reading and getting data from a list of .dat file from the get files function provides.\n
    Datafile must be a single .dat file.\n
    Start time must be in UTC e.g. '2017-03-09T00:00:00.000'.\n
    End time must be in UTC e.g. '2017-03-09T00:00:00.000'.\n
    """
    def __init__(self,dataFile,startTime,endTime):
        self.dataFileList = dataFile
        self.startTime = datetime.datetime.strptime(startTime,'%Y-%m-%dT%H:%M:%S') #Converted to datetime.datetime object for easier date manipulation
        self.endTime = datetime.datetime.strptime(endTime,'%Y-%m-%dT%H:%M:%S')
        self.dataDict = {}
        

    def getIonData(self):
        for dataFile in self.dataFileList:
            labelPath = dataFile.rstrip('.sts') + '.lbl'    #All .dat files should come with an accompanying .lbl file
            label = PDS3Label(labelPath)    #The label file is parsed for the data needed
            logging.debug(label.dataNameDict)
            rows = label.rows #All LRS jade data has 8640 rows of data per file
            species = 3 #The ion species interested in as defined in the label
            with open(dataFile, 'rb') as f:
                for _ in range(rows):
                    data = f.read(label.bytesPerRow)    
                    
                    timeData = label.dataNameDict['DIM0_UTC']   #Label data for the time stamp
                    startByte = timeData['START_BYTE']  #Byte where the time stamp starts
                    endByte = timeData['END_BYTE']  #Byte where the time stamp ends
                    dataSlice = data[startByte:endByte] #The slice of data that contains the time stamp
                    dateTimeStamp = datetime.datetime.strptime(str(dataSlice,'ascii'),'%Y-%jT%H:%M:%S.%f')  #The time stamp is converted from DOY format to a datetime object
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

                    if 'DIM1_ARRAY' not in self.dataDict[dateStamp]:
                        self.dataDict[dateStamp]['DIM1_ARRAY'] = dataArray

                    if ionSpecies == species:   #If the species for the row is the desired species continue finding data
                        
                        if 'TIME_ARRAY' not in self.dataDict[dateStamp]:
                            self.dataDict[dateStamp]['TIME_ARRAY'] = []
                        self.dataDict[dateStamp]['TIME_ARRAY'].append(timeStamp)   #Array to hold time stamps is created and the decimal hour time is appended to it

                        if 'DATETIME_ARRAY' not in self.dataDict[dateStamp]:
                            self.dataDict[dateStamp]['DATETIME_ARRAY'] = []
                        self.dataDict[dateStamp]['DATETIME_ARRAY'].append(str(dateTimeStamp))   #Array to hold time stamps is created and the decimal hour time is appended to it
                            
                        dataObjectData = label.dataNameDict['DATA'] #Label data for the data is found 
                        startByte = dataObjectData['START_BYTE']
                        endByte = dataObjectData['END_BYTE']
                        dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                        temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                        temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                        dataArray = [np.mean(row) for row in temp]  #Each rows average is found to have one column 

                        if 'DATA_ARRAY' not in self.dataDict[dateStamp]:
                            self.dataDict[dateStamp]['DATA_ARRAY'] = []
                        
                        self.dataDict[dateStamp]['DATA_ARRAY'].append(np.log(dataArray)) #The log of the data column is taken and appended to the data dictionary under the key DATA_ARRAY

                        latObjectData = label.dataNameDict['SC_POS_LAT'] #Label data for the data is found 
                        startByte = latObjectData['START_BYTE']
                        endByte = latObjectData['END_BYTE']
                        dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                        latArray = struct.unpack(latObjectData['FORMAT']*latObjectData['DIM1'],dataSlice)[0] #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once

            f.close()

    def getElecData(self):
        for dataFile in self.dataFileList:
            labelPath = dataFile.rstrip('.DAT') + '.LBL'    #All .dat files should come with an accompanying .lbl file
            label = PDS3Label(labelPath)    #The label file is parsed for the data needed
            logging.debug(label.dataNameDict)
            rows = label.rows #All LRS jade data has 8640 rows of data per file
            with open(dataFile, 'rb') as f:
                for _ in range(rows):
                    data = f.read(label.bytesPerRow)    
                    
                    timeData = label.dataNameDict['DIM0_UTC']   #Label data for the time stamp
                    startByte = timeData['START_BYTE']  #Byte where the time stamp starts
                    endByte = timeData['END_BYTE']  #Byte where the time stamp ends
                    dataSlice = data[startByte:endByte] #The slice of data that contains the time stamp
                    dateTimeStamp = datetime.datetime.strptime(str(dataSlice,'ascii'),'%Y-%jT%H:%M:%S.%f')  #The time stamp is converted from DOY format to a datetime object
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
                    
                    if 'TIME_ARRAY' not in self.dataDict[dateStamp]:
                            self.dataDict[dateStamp]['TIME_ARRAY'] = []
                    self.dataDict[dateStamp]['TIME_ARRAY'].append(timeStamp)   #Array to hold time stamps is created and the decimal hour time is appended to it

                    if 'DATETIME_ARRAY' not in self.dataDict[dateStamp]:
                        self.dataDict[dateStamp]['DATETIME_ARRAY'] = []
                    self.dataDict[dateStamp]['DATETIME_ARRAY'].append(str(dateTimeStamp))   #Array to hold time stamps is created and the decimal hour time is appended to it
                            
                    dataObjectData = label.dataNameDict['DATA'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [np.mean(row) for row in temp]  #Each rows average is found to have one column 

                    if 'DATA_ARRAY' not in self.dataDict[dateStamp]:
                        self.dataDict[dateStamp]['DATA_ARRAY'] = []
                    
                    self.dataDict[dateStamp]['DATA_ARRAY'].append(np.log(dataArray)) #The log of the data column is taken and appended to the data dictionary under the key DATA_ARRAY

                    dataObjectData = label.dataNameDict['DIM1_E'] #Label data for the data is found 
                    startByte = dataObjectData['START_BYTE']
                    endByte = dataObjectData['END_BYTE']
                    dataSlice = data[startByte:endByte] #Slice containing the data for that row is gotten
                    temp = struct.unpack(dataObjectData['FORMAT']*dataObjectData['DIM1']*dataObjectData['DIM2'],dataSlice) #The binary format of the data is multiplied by the dimensions to allow unpacking of all data at once
                    temp = np.asarray(temp).reshape(dataObjectData['DIM1'],dataObjectData['DIM2'])  #The data is put into a matrix of the size defined in the label
                    dataArray = [row[0] for row in temp]  #Each rows average is found to have one column 


                    if 'DIM1_ARRAY' not in self.dataDict[dateStamp]:
                        self.dataDict[dateStamp]['DIM1_ARRAY'] = dataArray

                    
#--------------------------------------------------------------------------------------
                        
                        
def test():
    
    data_list = ['/home/delamere/Downloads/jno219/fgm_jno_l3_2017219pc_r1s_v01.sts']
    start_time = '2017-08-07T00:00:00'
    end_time = '2017-08-07T23:59:59'
    
    fgm_data = FGMData(data_list,start_time,end_time)
    fgm_data.getIonData()
    print(fgm_data.dataDict)
    
    
    
if __name__ == '__main__':
    test()