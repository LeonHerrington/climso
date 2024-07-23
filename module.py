
import socket
from astropy.io import fits

import astropy.units as u 
from astropy.table import QTable

from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames, RotatedSunFrame
from sunpy.map.header_helper import make_heliographic_header

from skimage import measure

from ftplib import FTP
from sunpy.io.special import srs



def getHeader(hdu):
    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                 observer='earth', frame=frames.Helioprojective)
    
    scale  = [0.5*(hdu.header['NAXIS1']-100)/hdu.header['RSUN_OBS'], 0.5*(hdu.header['NAXIS2']-100)/hdu.header['RSUN_OBS']]
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[1024, 1024]*u.pixel,
                                        scale=scale*u.arcsec/u.pixel)
    
    header['rsun_obs'] = hdu.header['rsun_obs']
    
    return header
    
    
def toSunpyMap(filename, center_disk=False):

    with fits.open(filename) as hdul:

        if center_disk: hdul[0] = centerDisk(hdul[0])
        
        header = getHeader(hdul[0])
        
        hdul[0].data = np.flip(hdul[0].data, axis=0)
        
        map = sunpy.map.Map(hdul[0].data, header, map_type='generic_map')
        
    return map


def carrington(filename, flat=False, center=False):
    
    map = toSunpyMap(filename, center_disk=center)
    
    # removes outer pixel
    map.meta['cdelt1'] *= 1.01
    map.meta['cdelt2'] *= 1.01
    
    if flat:
        weights, _ = getWeights(map)
        weights[np.isnan(weights)]=1
        flattened = map.data + 0.8*np.median(map.data)*weights
        map = sunpy.map.Map(flattened, map.meta)

    carr_header = make_heliographic_header(map.date, map.observer_coordinate, map.data.shape, frame='carrington')
    
    outmap = map.reproject_to(carr_header)

    return outmap


def getWeights(map):
    coordinates = sunpy.map.all_coordinates_from_map(map)
    weights     = coordinates.transform_to("heliocentric").z.value

    mu = np.array(weights / np.nanmax(weights))

    weights = np.ones(mu.shape) - mu
    
    return weights, mu

def flatten(map):
    weights, _ = getWeights(map)
    
    weights[np.isnan(weights)]=1
    
    flattened = map.data + 0.7*np.mean(map.data[map.data>10000])*weights # was 0.75
    
    return flattened


def getUmbraPenumbra(map):
    
    # Flatten (limb darkness correction)
    flattened = flatten(map)
    
    mask = np.zeros(map.data.shape, dtype=np.uint8)
    cv.circle(mask, (int(map.meta['crpix1']), int(map.meta['crpix2'])), int(map.meta['rsun_obs']-20), 1, thickness=-1)

    flattened[mask==0]=np.nan
    
    # threshold
    _, umbra    = cv.threshold(flattened,0.7*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)
    _, sunspot  = cv.threshold(flattened,0.91*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)
    _, penumbra = cv.threshold(flattened,0.95*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)

    # Morph
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    umbra  = cv.morphologyEx(umbra.astype(np.uint8), cv.MORPH_OPEN, kernel)

    kernel  = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    sunspot = cv.morphologyEx(sunspot.astype(np.uint8), cv.MORPH_OPEN, kernel)

    kernel   = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    penumbra = cv.morphologyEx(penumbra.astype(np.uint8), cv.MORPH_OPEN, kernel)
    
    # Removes bad penumbra using sunspot
    n_labels, labels = cv.connectedComponents(penumbra, connectivity=8)

    keep_label_list = np.unique(cv.bitwise_and(labels,sunspot.astype(np.int32)))
    mask = np.isin(labels, keep_label_list)

    penumbra[~mask] = 0
    labels[~mask]   = 0

    # turns penumbra with no umbra into umbra
    remove_label_list = np.unique(cv.bitwise_and(labels,umbra.astype(np.int32)))
    mask = np.isin(labels, remove_label_list)

    umbra[~mask]=255

    penumbra = penumbra - umbra

    
    return umbra, penumbra
    
    
def drawSunspots(map, umbra=None, penumbra=None):
    
    if umbra==None or penumbra==None:
        umbra, penumbra = getUmbraPenumbra(map)
    
    img = cv.convertScaleAbs(map.data, alpha=(255.0/65535.0)).astype(np.uint8)
    color_image = np.zeros((umbra.shape[0], umbra.shape[1], 3), dtype=np.uint8)
    color_image[umbra == 255]    = [0, 0, 255] 
    color_image[penumbra == 255] = [255, 0, 0] 

    img_label = cv.addWeighted(cv.cvtColor(img, cv.COLOR_GRAY2RGB),2,color_image,0.2,0)

    contours, hierarchy = cv.findContours(umbra, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_label, contours, -1, (0,0,255), 1)

    contours, hierarchy = cv.findContours(penumbra, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_label, contours, -1, (255,0,0), 1)
    
    return img_label


def groupSunspots(map, threshold=0.01) -> QTable:
    
    umbra, penumbra = getUmbraPenumbra(map)
    
    # Label
    n_labels, labels, _, centroids = cv.connectedComponentsWithStats(umbra | penumbra, connectivity=8)
    centroids=centroids[1:]
    
    groups = []
    for i, centroid1 in enumerate(centroids):
        point1 = map.pixel_to_world(centroid1[0]*u.pixel, (centroid1[1])*u.pixel)
        group_found = False
        for group in groups:
            for _, centroid2 in group:
                point2   = map.pixel_to_world(centroid2[0]*u.pixel, (centroid2[1])*u.pixel)
                distance = point1.separation(point2).value 
                if distance < threshold:
                    group.append((i+1, centroid1))
                    group_found = True
                    break
            if group_found:
                break
        if not group_found:
            groups.append([(i+1, centroid1)])
    
    # Assign new label
    label_image = np.zeros(labels.shape, dtype=np.uint16)

    for idx, group in enumerate(groups):
        for label, centroid in group:
            label_image[labels==label]=idx+1
    
    # Centroids  
    regions = measure.regionprops(label_image)

    centroids_groups = np.array([[r.centroid[1], r.centroid[0]] for r in regions]).astype(np.uint16)
    
    lon = []
    lat = []
    x   = []
    y   = []
    for centroid in centroids_groups:
        point = map.pixel_to_world(centroid[0]*u.pixel, centroid[1]*u.pixel).heliographic_stonyhurst
        lon.append(point.lon.deg)
        lat.append(point.lat.deg)
        x.append(centroid[0])
        y.append(centroid[1])
        
    table = QTable(
                [np.arange(1,len(lon)+1).astype(np.uint16), [np.ma.masked]*len(lon), [np.ma.masked]*len(lon), lon*u.deg, lat*u.deg, x*u.pixel, y*u.pixel],
                names=('label', 'id', 'noaa', 'longitude', 'latitude', 'x', 'y'),
                meta={'date': map.date},
                )
    
    return table


# SRS

def getSRSTable(date) -> QTable:
    
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
    

# assign NOAA
def assignNOAAToTable(table, map) -> QTable:
    srs_table = getSRSTable(map.date)
    
    date_obs = srs_table.meta['issued'].replace(hour=0, minute=0)
    
    for region in srs_table[srs_table['ID']=='I']:
        point_noaa = SkyCoord(region['Longitude'],
                    region['Latitude'], 
                    obstime=date_obs, 
                    observer='earth', 
                    frame=frames.HeliographicStonyhurst,
                    )

        diffrot_point = SkyCoord(RotatedSunFrame(base=point_noaa, rotated_time=map.date))
        transformed_diffrot_point = diffrot_point.transform_to(map.coordinate_frame)
        
        min_dist = np.inf
        for idx, centroid_old in enumerate(table):
            point_old = map.pixel_to_world(centroid_old['x'], centroid_old['y'])
            dist      = transformed_diffrot_point.separation(point_old).deg
            
            if min_dist > dist:
                min_dist = dist
                min_idx  = idx
        
        if min_dist<0.02:
            table[min_idx]['noaa'] = region['Number']
            
    return table



# Utilities ##################################################


import cv2 as cv
import numpy as np



def centerDisk(hdu):
    _, disk = cv.threshold(hdu.data, np.median(hdu.data)/2, 255, cv.THRESH_BINARY)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(100,100))
    disk   = cv.morphologyEx(disk.astype(np.uint8), cv.MORPH_OPEN, kernel)
    
    disk=disk.astype(np.uint8)
    
    contours, _ = cv.findContours(disk, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Assuming the disk is the largest contour
    largest_contour = max(contours, key=cv.contourArea)
    
    # Get radius
    (x_axis,y_axis), radius = cv.minEnclosingCircle(largest_contour) 
    hdu.header['rsun_obs'] = radius # sun radius in pixels

    # Calculate translation required to center the centroid
    rows, cols    = disk.shape
    center_x      = cols // 2
    center_y      = rows // 2
    translation_x = center_x - int(x_axis)
    translation_y = center_y - int(y_axis)

    # Translate the image
    hdu.data = np.roll(hdu.data, translation_x, axis=1)
    hdu.data = np.roll(hdu.data, translation_y, axis=0)
    
    return hdu


# Misc ##################################################


import os

def getMostRecent(directory, contains_string):
    # Get list of folders in the directory
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    
    # Sort folders by modification time (most recent first)
    folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not folders:
        print("No folders found in the directory")
        return None
    
    # Get list of files in the most recent folder
    for most_recent_folder in folders:
        files_in_folder = [f.path for f in os.scandir(most_recent_folder) if f.is_file() and contains_string in f.name]
    
        if files_in_folder:
            break
    
    # Sort files by modification time (most recent first)
    files_in_folder.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files_in_folder[0]


# SunspotIndex

def get_current_index():
    try:
        with open("index.txt", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def save_current_index(index):
    with open("index.txt", "w") as f:
        f.write(str(index))

def get_new_id():
    current_index = get_current_index()
    new_id = current_index + 1
    save_current_index(new_id)
    return new_id

