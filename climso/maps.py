import os
import re

import numpy as np
import sunpy.map
import sunpy.coordinates
import matplotlib.pyplot as plt

from .utils import toSunpyMap, medianFlatten

### Classes:

class SynopticMap:
    """
    Generates a synoptic map from CLIMSO data.

    Parameters:
    ------    
    climso_dir (str) : directory containing the CLIMSO data (ex:'/data/CLIMSO/').
    
    rotation (int) : carrington rotation number to use for the synoptic map.
    
    disk (str) : type of disk to use 'l1' or 'l2'. Default is 'l1'.
    
    methods
    ------    
    data : synoptic map array.
        
    plot() : plots the synopticMap.
    
    """

    def __init__(self, climso_dir:str, rotation:int, disk='l1'):
        """
        Initializes the SynopticMap class.

        Parameters
        ------        
        climso_dir (str): directory containing the CLIMSO data (ex:'/data/CLIMSO/').
        
        rotation (int): carrington rotation number to use for the synoptic map. default is last rotation.
        
        disk (str): type of disk to use 'l1' or 'l2'. Default is 'l1'.
        """
        
        self.rotation=rotation
        self.disk=disk
        self.data = getSynopticMap(climso_dir, rotation, disk)
        
    
    def plot(self,figsize=(10,5), cmap='gray', vmin=None, vmax=None):
        """
        Plots the synopticMap.
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.data, origin='lower', extent=[0,360,-90,90], cmap=cmap, vmin=vmin, vmax=vmax)
 
        plt.xticks(np.arange(0, 361, 30))
        plt.yticks(np.arange(-90,91,30))
        
        plt.axis()

        plt.xlabel('Carrington Longitude');
        plt.ylabel('Latitude');


### Functions:

def getSynopticMap(climso_dir, rotation: int, disk='l1'):
    """
    Generates a synoptic map from CLIMSO data.

    Parameters
    ------    
    climso_dir (str): directory containing the CLIMSO data (ex:'/data/CLIMSO/').
    
    rotation (int): carrington rotation number to use for the synoptic map.
    
    disk (str): type of disk to use 'l1' or 'l2'. Default is 'l1'.
    """        
    
    ### Get files
    start_time = sunpy.coordinates.sun.carrington_rotation_time(rotation)
    end_time = sunpy.coordinates.sun.carrington_rotation_time(rotation+1)
    
    start = int(start_time.strftime("%Y%m%d"))
    end = int(end_time.strftime("%Y%m%d"))
    
    files = []
    for year in range(start_time.datetime.year, end_time.datetime.year + 1):
        year_folder = os.path.join(climso_dir, str(year),'data_calibrated')
        if os.path.isdir(year_folder):
            for date_folder in os.listdir(year_folder):            
                    folder_date = int(date_folder.replace('-', ''))
                    if start <= folder_date <= end:
                        date_folder = os.path.join(year_folder, date_folder)
                        if os.path.isdir(date_folder):
                            closest_time=np.inf
                            file=None
                            for filename in os.listdir(date_folder):
                                time = re.findall(fr'_{disk}_.{{8}}_(\d+)_emi1.fts', filename)
                                if len(time) > 0:
                                    file_time = int(time[0])
                                    if abs(file_time-11000000)<closest_time:
                                        closest_time = file_time
                                        file = filename
                            if file: files.append(os.path.join(date_folder, file))
                            
    if len(files)<2: 
        print('Not enough images found')
        return None
    
    ### Carrington projections
    carrington_list = []
    for filename in files:
        carrington_list.append(carrington(filename, flat=True, center=True, mean=30000))
    
    carrington_list.sort(key=lambda x: x.carrington_longitude.deg)
    
    carr_longs = []
    for carr in carrington_list:
        idx = int((carr.carrington_longitude.deg)*carr.data.shape[1]/360)
        carr_longs.append(idx)
    
    ### Merge into synoptic map
    synoptic_map = np.full(carrington_list[0].data.shape, np.nan)

    for idx in range(len(carrington_list)-1):
        
        left  = np.roll(carrington_list[idx].data, 1024, axis = 1)
        right = np.roll(carrington_list[idx+1].data, 1024, axis = 1)
        
        line = np.concatenate((
            np.full((carr_longs[idx]),np.nan),
            np.linspace(1,0, carr_longs[idx+1]-carr_longs[idx]),
            np.full((2048-carr_longs[idx+1]),np.nan),
            ))
        
        left_nan = np.isnan(left[:,carr_longs[idx]:carr_longs[idx+1]])
        right_nan = np.isnan(right[:,carr_longs[idx]:carr_longs[idx+1]])
        
        mask_left = np.tile(line, (synoptic_map.shape[0],1))
        mask_left[:,carr_longs[idx]:carr_longs[idx+1]][right_nan]=1
        
        mask_right = np.tile(1-line, (synoptic_map.shape[0],1))
        mask_right[:,carr_longs[idx]:carr_longs[idx+1]][left_nan]=1
        
        synoptic_map[:,carr_longs[idx]:carr_longs[idx+1]] = np.nansum([left*mask_left, right*mask_right], axis=0)[:,carr_longs[idx]:carr_longs[idx+1]]
        synoptic_map[:,carr_longs[idx]:carr_longs[idx+1]][left_nan & right_nan]=np.nan
        
    # Roll again to put back    
    synoptic_map = np.roll(synoptic_map, 1024, axis = 1)

    # Last and first carrington case ------------------------------------------------------------------------/
    left  = carrington_list[-1].data
    right = carrington_list[0].data

    # undoing rotation
    left_long = carr_longs[-1]+1024
    right_long = carr_longs[0]+1024

    # dealing with edge cases
    if left_long>=2048     : left_long  = left_long  - 2048
    if right_long>=2048    : right_long = right_long - 2048
    if left_long>right_long: left_long  = left_long  - 2048


    line = np.full((1,2048), np.nan)

    if left_long<0:
        line[:,left_long:]=np.linspace(1,0, right_long-left_long)[:-left_long]
        line[:,:right_long]=np.linspace(1,0, right_long-left_long)[-left_long:]
        
        mask_left = np.tile(line, (synoptic_map.shape[0],1))
        mask_left[:,left_long:][np.isnan(right[:,left_long:])]   = 1
        mask_left[:,:right_long][np.isnan(right[:,:right_long])] = 1
        
        mask_right = np.tile(1-line, (synoptic_map.shape[0],1))
        mask_right[:,left_long:][np.isnan(left[:,left_long:])]   = 1
        mask_right[:,:right_long][np.isnan(left[:,:right_long])] = 1
        
        synoptic_map[:,left_long:]  = np.nansum([left*mask_left, right*mask_right, synoptic_map], axis=0)[:,left_long:]
        synoptic_map[:,:right_long] = np.nansum([left*mask_left, right*mask_right, synoptic_map], axis=0)[:,:right_long]
        
        synoptic_map[:,left_long:][np.isnan(left[:,left_long:]) & np.isnan(right[:,left_long:])]    = np.nan
        synoptic_map[:,:right_long][np.isnan(left[:,:right_long]) & np.isnan(right[:,:right_long])] = np.nan

    else:
        line[:,left_long:right_long]=np.linspace(1,0, right_long-left_long)
        
        mask_left = np.tile(line, (synoptic_map.shape[0],1))
        mask_left[:,left_long:right_long][np.isnan(right[:,left_long:right_long])]=1
        
        mask_right = np.tile(1-line, (synoptic_map.shape[0],1))
        mask_right[:,left_long:right_long][np.isnan(left[:,left_long:right_long])]=1
        
        synoptic_map[:,left_long:right_long] = np.nansum([left*mask_left, right*mask_right], axis=0)[:,left_long:right_long]
        synoptic_map[:,left_long:right_long][np.isnan(left[:,left_long:right_long]) & np.isnan(right[:,left_long:right_long])]=np.nan
    
    # roll to make 0 to 360 instead of -180 to 180
    synoptic_map = np.roll(synoptic_map, 1024, axis = 1)

    return synoptic_map


def carrington(filename, flat=False, center=False, mean=None):
    """
    Parameters
    ------
    filename (str) : FITS file (disk image : l1 or l2).
    
    flat (bool) : If true flattens image to correct limb darkening and other effects.
    
    center (bool) : If true centers the disk.
    
    mean (int) : Sets the mean value of the image. Unchanged if left empty.
    
    Returns
    ------
    Carrington map
    """
    
    map = toSunpyMap(filename, center_disk=center)
    
    # removes outer pixel
    map.meta['cdelt1'] *= 1.01
    map.meta['cdelt2'] *= 1.01
    
    if flat:
        flattened = medianFlatten(map)
        
        if (mean is None) :
            map = sunpy.map.Map(flattened, map.meta)
        else:
            map = sunpy.map.Map(flattened + (mean - np.mean(flattened)), map.meta)
            
    elif (mean is not None):
        map = sunpy.map.Map(map.data + (mean - np.median(map.data)), map.meta)

    carr_header = sunpy.map.header_helper.make_heliographic_header(map.date, map.observer_coordinate, map.data.shape, frame='carrington')
    
    carrington_map = map.reproject_to(carr_header)

    return carrington_map