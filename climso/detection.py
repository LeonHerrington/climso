import math, socket
from ftplib import FTP
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import cv2 as cv
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames, RotatedSunFrame
from sunpy.io.special import srs
from skimage import measure
import matplotlib.pyplot as plt

from .utils import medianFlatten, get_mu


## Active regions ######################################################################################

class ActiveRegions:
    """
    Parameters
    ------
    map (sunpy.map.GenericMap) : disk map (l2)
    
    """
    
    def __init__(self, map):
        
        srs_table = self.getSRSTable(map.date)
        
        self.map = map
        self.active_regions = self.getAR(map)
        self.labels, self.table = self.groupNOAA(map, active_regions=self.active_regions, srs_table=srs_table)
        
    
    def getSRSTable(self, date:datetime) -> QTable:
        """
        Parameters
        -------
        date (datetime): date of the map
        
        Returns
        ------- 
        QTable : NOAA table containing active regions information that is released every day a bit after 00:00 UT
        """
        
        filename = 'pub/warehouse/' + str(date.datetime.year) + '/SRS/' + date.strftime("%Y%m%d") + 'SRS.txt'

        while True:
            try:
                with FTP('ftp.swpc.noaa.gov') as ftp:
                    ftp.login()
                    
                    file_contents = []
                    ftp.retrlines('RETR '+filename, file_contents.append)
                    
                    header, section_lines, supplementary_lines = srs.split_lines(file_contents)
                    srs_table = srs.make_table(header, section_lines, supplementary_lines)
                    
                    return srs_table
                
            except socket.gaierror:
                print('[Errno 11001] getaddrinfo failed : Trying again')
            
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
    
    
    def getAR(self, map, threshold=0.82) -> np.ndarray[np.dtype[np.uint8]]:
        """
        Parameters
        ------
        map (sunpy.map.GenericMap)
        
        threshold (float) : proportion of image median value to use as threshold (higher will over detect, lower will under detect)
        
        Returns
        ------
        Active Regions (ndarray[dtype[uint8], Any]) : array value of 255 is an active region, the rest is 0.
        """
        
        # Flatten (limb darkness correction)
        flattened = medianFlatten(map)
            
        mask = np.zeros(map.data.shape, dtype=np.uint8)
        cv.circle(mask, (1024,1024), int(map.meta['rsun_obs']-20), 1, thickness=-1)

        flattened[mask==0]=np.nan
                
        # threshold
        thresh = threshold*np.nanmedian(flattened)
        _, active_regions = cv.threshold(flattened,thresh,255,cv.THRESH_BINARY_INV)
        active_regions    = active_regions.astype(np.uint8)
        
        # Morph
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        active_regions = cv.morphologyEx(active_regions.copy(), cv.MORPH_OPEN, kernel)
        
        return active_regions


    def groupNOAA(self, map, active_regions, srs_table:QTable, distance_threshold=0.05) -> tuple[np.ndarray[np.dtype[np.int32]], QTable]:
        """
        Parameters
        ------
        map (sunpy.map.GenericMap)
        
        active_regions (ndarray) : array from getAR().
        
        srs_table (QTable) : from getSRSTable().
        
        distance_threshold (float) : maximum distance to group regions
        
        Returns
        ------
        labels, table (ndarray[np.dtype[np.int32], QTable) : array where value indicates region label, 0 is no region and table containing NOAA region numbers, coordinates and areas.
        
           
        """
        
        # SRS table to map differential rotation
        date_obs_srs = srs_table.meta['issued'].replace(hour=0, minute=0)
        
        noaa_list=[]
        for region in srs_table[srs_table['ID']=='I']:
            point_noaa = SkyCoord(region['Longitude'],
                    region['Latitude'],
                    obstime=date_obs_srs,
                    observer='earth',
                    frame=frames.HeliographicStonyhurst,
                    )
            diffrot_point = SkyCoord(RotatedSunFrame(base=point_noaa, rotated_time=map.date))
            transformed_diffrot_point = diffrot_point.transform_to(map.coordinate_frame)
            
            noaa_list.append((region['Number'], transformed_diffrot_point))
        
        
        # Label
        _, labels, _, centroids = cv.connectedComponentsWithStats(active_regions, connectivity=8)
        centroids=centroids[1:]
        
        for i, centroid in enumerate(centroids):
            point = map.pixel_to_world(centroid[0]*u.pixel, (centroid[1])*u.pixel)
            
            min_dist = np.inf
            for noaa_num, pointNOAA in noaa_list:
                dist = point.separation(pointNOAA).deg
                if dist < min_dist:
                    min_dist = dist
                    min_noaa = noaa_num

            if min_dist < distance_threshold:
                labels[labels==i+1]=min_noaa


        # Centroids  
        regions = measure.regionprops(labels)
        
        radius = map.meta['RSUN_OBS']
        mu = get_mu(map)
        
        num  = []
        lat  = []
        lon  = []
        x    = []
        y    = []
        area = []
        
        for r in regions:
            point = map.pixel_to_world(r.centroid[1]*u.pixel, r.centroid[0]*u.pixel).heliographic_stonyhurst
            lat.append(round(point.lat.deg, 2))
            lon.append(round(point.lon.deg, 2))
            
            num.append(r.label)
            x.append(int(r.centroid[1]))
            y.append(int(r.centroid[0]))
            
            area.append(round((np.sum(labels == r.label) / (2*np.pi*radius**2)) * (10**6 / mu[int(round(r.centroid[1])), int(round(r.centroid[0]))]),2))
            
            
        uSH = u.def_unit('uSH')
        table = QTable(
                    [
                        num,
                        lat *u.deg,
                        lon *u.deg,
                        x   *u.pixel,
                        y   *u.pixel,
                        area*uSH
                    ],
                    names=('number', 'latitude', 'longitude', 'x', 'y', 'area'),
                    meta={'date': map.date},
                    )
        
        return labels, table
        
    def plot(self, figsize=(8,8), cmap='gray'):
        """
        Plots active regions.
        """
        plt.figure(figsize=figsize)
        
        image = cv.convertScaleAbs(self.map.data, alpha=(255.0/65535.0)).astype(np.uint8)

        # Colors
        color = (255,0,0)
        weight = np.max(image)/255.0
        
        color_image = np.zeros((self.active_regions.shape[0], self.active_regions.shape[1], 3), dtype=np.uint8)
        color_image[self.active_regions==255] = [255,0,0]

        img_label = cv.addWeighted(cv.cvtColor(image, cv.COLOR_GRAY2RGB), 1/weight, color_image, weight,0)

        contours, _ = cv.findContours(self.active_regions, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img_label, contours, -1, color, 1)

        # Numbers
        font = cv.FONT_HERSHEY_DUPLEX
        fontScale = 1.2
        thickness = 1

        img_num = img_label.copy()
        for ar in self.table :
            cv.putText(img_num, str(ar['number']), (int(ar['x'].value) + 10, int(ar['y'].value) + 10), font, fontScale, color, thickness, cv.LINE_AA, True)

        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(img_num, origin='lower', cmap=cmap);
        

## Prominences #########################################################################################

class Prominences:

    def __init__(self, filename):
        
        self.file = filename
        self.array, self.date_obs = self.radialArray(self.filename)
        self.prominences = self.getProminences(self.filename, self.array)
    
    
    def radialArray(self,filename) -> np.ndarray[np.dtype[np.float64]]:
        """
        Parameters
        ------
        filename (str) : Fits c1 or c2 image filename.
        
        Returns
        ------
        radial_array (ndarray), date_obs (datetime)
        """
        radial_array = np.full((301, 720),fill_value=np.nan)
        
        # Load file
        with fits.open(filename) as hdul:
            
            data = np.flip(hdul[0].data, axis=0)
            date_obs = datetime.strptime(hdul[0].header['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
            
            # pre calculating sin cos values
            sincos = []
            stepSize = math.pi/360.0
            t0 = math.pi/2.0
            t = t0
            while t < 2*math.pi+t0:
                sincos.append((math.sin(t), math.cos(t)))
                t += stepSize
            
            # circle to array
            for idx, r_coeff in enumerate(np.arange(1,1.3,1e-3)):
                
                r = hdul[0].header['rsun_obs']*r_coeff
                if r > 1023 : break
                circle = []
                for sin, cos in sincos:
                    circle.append((round(r * sin + 1024), round(r * cos + 1024)))
                    t += stepSize
                
                line = [data[coord] for coord in circle]
                radial_array[idx, :] = line
        return radial_array, date_obs

    def getProminencePositions(self, filename, threshold=20000, r_coeff = 1.03) -> Optional[list[int]]:
        """
        Parameters
        ------
        filename (str) : Fits c1 or c2 image filename.
        threshold (int) : Intensity value above which a region in the image is considered a prominence.
        r_coeff (float) : Solar radius at which to detect prominences
        
        Returns
        ------
        list[int] : list of indices representing the central position of each prominence in degrees.
        """
        
        with fits.open(filename) as hdul:
            data = np.flip(hdul[0].data, axis=0)
                    
            r = hdul[0].header['rsun_obs']*r_coeff

        if r > 1023 : return None
        
        # Generate circle
        circle = []
        stepSize = math.pi/360.0        
        t0 = math.pi/2.0
        t = t0
        while t < 2 * math.pi+t0:
            circle.append((round(r * math.sin(t) + 1024), round(r * math.cos(t) + 1024)))
            t += stepSize

        # Get line from circle
        line = [data[coord] for coord in circle]

        # Find indices where line values exceed the threshold
        indices = np.where(np.array(line) > threshold)[0]

        # group indices into continuous regions considering the circular nature
        groups = []
        current_group = [indices[0]]
        length = len(line)

        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1 or (indices[i] == 0 and indices[i-1] == length - 1):
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]
        groups.append(current_group)

        # Handle circular connection between last and first group
        if groups[0][0] == 0 and groups[-1][-1] == length - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

        # Calculate the central indice for each prominence group
        central_indices = [group[len(group)//2]//2 for group in groups]

        return central_indices
    
    def maxRadius(self, radial_array, deg:float, threshold=10000) -> Optional[dict]:
        """
        Parameters
        ------
        radial_array (ndarray) : From radialArray().
        
        deg (float) : Position to search for maximum radius.
        
        threshold (int) : Threshold intensity to be considered a prominence.
        
        Returns
        ------
        Maximum radius (float) : in solar radiuses.
        """
        maximums = []
        deg_step = 2
        
        for idx, line in enumerate(radial_array):
            gap=0
            
            while gap<4:
                gap+=1
                h_scaling = int(6.0*idx/radial_array.shape[0])
                filtered_line = line[int(deg*deg_step)-(gap+h_scaling):int(deg*deg_step)+(gap+h_scaling)]
                line_max = np.nanmax(filtered_line)
                if line_max > threshold :
                    maximums.append({
                        'deg': (deg - gap/deg_step) + np.argmax(filtered_line)/deg_step,
                        'radius': (1 + idx*1e-3),
                        'value': line_max,
                    })
                    break
        
        if maximums==[]:
            return None
        
        max_radius_entry = max(maximums, key=lambda x:x['radius'])
        
        return max_radius_entry
  
  
    def getProminences(self, file, array) -> pd.DataFrame:
        """
        Parameters
        ------
        
        filename (str) : Fits c1 or c2 image filename.
        
        array (ndarray) : From radialArray().
        
        Returns
        ------
        DataFrame : prominences ('deg', 'radius', 'value').
        """
        prom_degs = self.getProminencePositions(file)
        
        prominences = []
        for deg in prom_degs:
            max_radius_entry = self.maxRadius(array, deg)
            if max_radius_entry :
                prominences.append(max_radius_entry)

        return pd.DataFrame(prominences)
    
    



### Functions: #########################################################################################

# def estimateSpeed(files:list[str], deg:float):
#     """
#     Parameters
#     ------
#     files (list[str]) : list of FITS files (c1 or c2) from which to estimate the speed of the prominence.
    
#     deg (float) : position of the prominence in degrees.

#     Returns
#     ------
#     speeds(float) : instantaneous velocity each calculated between two images, 
#     radiuses(float) : height of the prominence in solar radiuses for each image,
#     times(datetime) : datetime of each image,
#     degrees(float) : position of the "peak" of the prominence in degrees for each image.
#     """
    
#     speeds   = []
#     radiuses = []
#     times    = []
#     degrees  = []

#     for n in range(1, len(files)):
#         array1, date_obs1 = radialArray(files[n-1])
#         array2, date_obs2 = radialArray(files[n])

#         max_radius_entry_1 = maxRadius(array1, deg)
#         if max_radius_entry_1 is None: continue
#         max_radius_entry_2 = maxRadius(array2, max_radius_entry_1['deg'])
#         if max_radius_entry_2 is None: continue
        
#         deg = int(max_radius_entry_2['deg'])

#         distance = (max_radius_entry_2['radius'] - max_radius_entry_1['radius'])*696e3 # km
#         time = abs(date_obs2 - date_obs1)
#         if time.seconds == 0 :
#             continue
        
#         speed = distance / time.seconds # km/s
        
#         speeds.append(speed)   
#         radiuses.append(max_radius_entry_1['radius'])
#         times.append(date_obs1)
#         degrees.append(max_radius_entry_1['deg'])

#     if max_radius_entry_2:
#         radiuses.append(max_radius_entry_2['radius'])
#         times.append(date_obs2)
#         degrees.append(max_radius_entry_2['deg'])
        
#     return speeds, radiuses, times, degrees