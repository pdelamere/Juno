#!/usr/bin/env python3

import sys, os, pathlib, re, datetime

def stitch():
    pic_path1 = pathlib.Path(r'/data/Python/jupiter/figures/orbit2')
    pic_path2 = pathlib.Path(r"/data/Python/jupiter/figures/jedi_pitchangle/plot_pa_pj1_2/plot_pa_pj1_2")
    
    save_path = 'pacombinedorbits/paorbit2'
    print(f'Got paths {pic_path1}\n{pic_path2}')
    file_dict = {}
    for parent,child,files in os.walk(pic_path1):
        for file in files:
            file_path = os.path.join(parent,file)
            
            try:
                p = re.compile(r'\d{7}_\d{4}')
                date_time = p.search(file).group()
                
                date = datetime.datetime.strptime(date_time,'%Y%j_%H%M').strftime('%Y-%jT%H')
                
            except:
                continue
            
            file_dict[date] = [file_path]
            print(date)
    
    for parent,child,files in os.walk(pic_path2):
        for file in files:
            file_path = os.path.join(parent,file)
            
            
            p = re.compile(r'\d{4}-\d{3}T\d{4}')
            date_time = p.search(file).group()
            print(date_time)
            date_temp = datetime.datetime.strptime(date_time,'%Y-%jT%H%M')
            if date_temp.hour%6 != 0:
                if date_temp.hour >= 0 and date_temp.hour < 6:
                    date = date_temp.strftime('%Y-%jT') + '0'
                elif date_temp.hour >= 6 and date_temp.hour < 12:
                    date = date_temp.strftime('%Y-%jT') + '12'
                elif date_temp.hour >= 12 and date_temp.hour < 18:
                    date = date_temp.strftime('%Y-%jT') + '18'
                elif date_temp.hour >= 18 and date_temp.hour < 24:
                    date = date_temp + datetime.timedelta(days=1) 
                    date = date.strftime('%Y-%jT') + '00'
            else: date = datetime.datetime.strptime(date_time,'%Y-%jT%H%M%S').strftime('%Y-%jT%H')
            print(date)
            
            if date in file_dict.keys():
                file_dict[date].append(file_path)
            
    for date,paths in file_dict.items():
        if len(paths) >= 2:
            os.system(f'mogrify -resize 850x1100 {paths[1]}')
            os.system(f'mogrify -resize 850x1100 {paths[0]}')
            os.system(f'convert {paths[0]} {paths[1]} -append /data/Python/jupiter/figures/{save_path}/jno_{date}.png')
            os.system(f'mogrify -resize 1100x1150 /data/Python/jupiter/figures/{save_path}/jno_{date}.png')
            print(f'Graph Made {date}')
if __name__ == '__main__':
    stitch()



