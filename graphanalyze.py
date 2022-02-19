#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:11:16 2020

@author: delamere
"""

#!/usr/local/anaconda3/bin/python python3

import os, pathlib, datetime, pyautogui, keyboard, mouse, time, subprocess
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image,ImageTk

def analyze():    
        
    pic_folder = pathlib.Path('../figures/orbit6')
    pics = os.listdir(pic_folder)
    pic_paths = [os.path.join(pic_folder,pic) for pic in pics] 
    pic_num = 0

    time_list = []
    pos_list = [0]
    for i in range(0,21600+1):
        time_list.append(datetime.datetime(1998,6,1,0)+datetime.timedelta(seconds=i))
    for i in range(1,len(time_list)):
        pos_list.append(i/len(time_list))

    time_list_6 = [time_list[i].hour + time_list[i].minute/60 + time_list[i].second/3600 for i in range(len(time_list))]
    time_list_12 = [i + 6 for i in time_list_6]
    time_list_18 = [i + 6 for i in time_list_12]
    time_list_24 = [i + 6 for i in time_list_18]
    
    p = subprocess.Popen(["display",pic_paths[pic_num]])
    
    print('please specify graph area')
    print('leftmost x-axis portion')
    mouse.wait('left')
    
    pos = pyautogui.position()
    graph_area = [pos.x]
    
    time.sleep(.5)

    print('rightmost x-axis portion')
    mouse.wait('left')
    pos = pyautogui.position()
    graph_area.append(pos.x)
    
    time.sleep(.1)

    print('Graph area defined')
    while True:
        #Sheath to Sphere using nums?

        if keyboard.is_pressed('esc'):
            p.kill()
            break
        
        if keyboard.is_pressed('q'):
            print('Select new graph area')

            print('leftmost x-axis portion')
            mouse.wait('left')
            pos = pyautogui.position()
            graph_area = [pos.x]
            
            time.sleep(.5)

            print('rightmost x-axis portion')
            mouse.wait('left')
            pos = pyautogui.position()
            graph_area.append(pos.x)
            time.sleep(.1)

            print('Graph area defined')

        if mouse.is_pressed('left'):
            win = pyautogui.getActiveWindow()
            mouse_pos = pyautogui.position()
            graph_len = graph_area[1] - graph_area[0]

            if mouse_pos.x >= graph_area[1] and mouse_pos.x <= win.left+win.width:
                print('Next image')
                pic_num += 1
                p.kill()
                p = subprocess.Popen(["C:\Program Files\Honeyview\Honeyview.exe",pic_paths[pic_num]])
                time.sleep(0.5)

            elif mouse_pos.x >= win.left and mouse_pos.x <= graph_area[0]:
                print('Previous image')
                pic_num -= 1
                p.kill()
                p = subprocess.Popen(["C:\Program Files\Honeyview\Honeyview.exe",pic_paths[pic_num]])
                time.sleep(0.5)

            if mouse_pos.x >= graph_area[0] and mouse_pos.x <= graph_area[1]:
                print('Would you like to record this point as a crossing? Press 1 for yes and 2 for no')

                while True:
                    if keyboard.is_pressed('1'):
                        record = True 
                        print('Yes')
                        break
                    elif keyboard.is_pressed('2'):
                        record = False
                        print('No')
                        break
                
                if record: pass
                elif not record: continue
                time.sleep(0.05)

                mouse_rel_pos = (mouse_pos.x - graph_area[0])/graph_len
                date_time_stamp = datetime.datetime.strptime(pic_paths[pic_num][-16:-4],'%Y%j_%H%M')
                record_date = date_time_stamp.date()
                print(record_date)
                closest_pos = min(range(len(pos_list)), key=lambda j: abs(pos_list[j]-mouse_rel_pos))
                
                if date_time_stamp.hour == 0:
                    graph_time_stamp = time_list_6[closest_pos]
                elif date_time_stamp.hour == 6:
                    graph_time_stamp = time_list_12[closest_pos]
                elif date_time_stamp.hour == 12:
                    graph_time_stamp = time_list_18[closest_pos]
                elif date_time_stamp.hour == 18:
                    graph_time_stamp = time_list_24[closest_pos]
                
                print('What type of crossing is this? 1-To sheath 2-To magnetosphere')
                while True:
                    if keyboard.is_pressed('1'):
                        record_type = 'Sheath' 
                        break
                    elif keyboard.is_pressed('2'):
                        record_type = 'Magnetosphere'
                        break
                
                temp_hr = int(graph_time_stamp)
                temp_min = int((graph_time_stamp * 60) % 60)
                temp_sec = int((graph_time_stamp * 3600) % 60)
                
                record_time = datetime.datetime.strptime(f'{temp_hr}:{temp_min}:{temp_sec}','%H:%M:%S').time()

                print(f'{record_type} crossing at {record_date} {record_time} recorded.')
                
                with open('jno_crossings.dat','a') as crossing:
                    crossing.writelines(f'\n{record_date} ,{record_time} ,{record_type}')
                crossing.close()
                print('Continue analyzing images')
                
      
    


if __name__ == '__main__':

    analyze()
    
