
import bz2
from astropy.io import fits

import astropy.units as u 

from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_heliographic_header

from skimage import measure


def readFitsBz2(path):
    decompressed_file = bz2.BZ2File(path)
    hdul = fits.open(decompressed_file)
    
    primary_hdu = fits.PrimaryHDU(data=hdul[0].data, header=hdul[0].header) 
    
    hdul.close()
    return primary_hdu


def getHeader(hdu):
    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                 observer='earth', frame=frames.Helioprojective)
    
    scale = [0.5*(hdu.header['NAXIS1']-100)/hdu.header['RSUN_OBS'], 0.5*(hdu.header['NAXIS2']-100)/hdu.header['RSUN_OBS']]
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[hdu.header['CRPIX1'], hdu.header['CRPIX2']]*u.pixel,
                                        scale=scale*u.arcsec/u.pixel)
    
    header['rsun_obs'] = hdu.header['rsun_obs']
    
    return header
    
    
def toSunpyMap(filename):

    hdu = readFitsBz2(filename)

    header = getHeader(hdu)
    
    hdu.data = centerDisk(hdu.data)
    
    return sunpy.map.Map(hdu.data, header, map_type='generic_map')


def carrington(filename, weights=None):
    
    hdu = readFitsBz2(filename)
    
    if type(weights) is np.ndarray:
        hdu.data = hdu.data * weights
    
    header = getHeader(hdu)

    map = sunpy.map.Map(hdu.data, header)

    carr_header = make_heliographic_header(map.date, map.observer_coordinate, hdu.data.shape, frame='carrington')

    outmap = map.reproject_to(carr_header)

    return outmap


def getWeights(map):
    coordinates = sunpy.map.all_coordinates_from_map(map)
    coordinates = sunpy.map.all_coordinates_from_map(map)
    weights = coordinates.transform_to("heliocentric").z.value

    mu = np.array(weights / np.nanmax(weights))

    weights = np.array(np.ones(mu.shape) - mu)
    
    return weights, mu

def flatten(map):
    weights, _ = getWeights(map)
    
    flattened = map.data + 1.4*np.mean(map.data)*weights
    
    return flattened


def getUmbraPenumbra(map):
    
    # Flatten (limb darkness correction)
    flattened = flatten(map)
    
    # threshold
    _, umbra = cv.threshold(flattened,0.7*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)
    _, sunspot = cv.threshold(flattened,0.91*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)
    _, penumbra = cv.threshold(flattened,0.96*np.nanmedian(flattened),255,cv.THRESH_BINARY_INV)

    # Morph
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    umbra = cv.morphologyEx(umbra.astype(np.uint8), cv.MORPH_OPEN, kernel)
    cv.circle(umbra, (1024,1024), int(map.meta['rsun_obs']-15),color=(0,0,0), thickness = 30);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    sunspot = cv.morphologyEx(sunspot.astype(np.uint8), cv.MORPH_OPEN, kernel)
    cv.circle(sunspot, (1024,1024), int(map.meta['rsun_obs']-15),color=(0,0,0), thickness = 30);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    penumbra = cv.morphologyEx(penumbra.astype(np.uint8), cv.MORPH_OPEN, kernel)
    cv.circle(penumbra, (1024,1024), int(map.meta['rsun_obs']-15),color=(0,0,0), thickness = 30);
    
    # Removes bad penumbra using sunspot
    n_labels, labels = cv.connectedComponents(penumbra, connectivity=8)

    keep_label_list = np.unique(cv.bitwise_and(labels,sunspot.astype(np.int32)))
    mask = np.isin(labels, keep_label_list)

    penumbra[~mask] = 0
    labels[~mask] = 0

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
    color_image[umbra == 255] = [0, 0, 255] 
    color_image[penumbra == 255] = [255, 0, 0] 

    img_label = cv.addWeighted(cv.cvtColor(img, cv.COLOR_GRAY2RGB),2,color_image,0.2,0)

    contours, hierarchy = cv.findContours(umbra, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_label, contours, -1, (0,0,255), 1)

    contours, hierarchy = cv.findContours(penumbra, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_label, contours, -1, (255,0,0), 1)
    
    return img_label

def groupSunspots(map, threshold=0.05):
    
    umbra, penumbra = getUmbraPenumbra(map)
    
    # Label
    n_labels, labels, _, centroids = cv.connectedComponentsWithStats(umbra | penumbra, connectivity=8)

    unique_labels = np.arange(1,n_labels)

    for label in unique_labels:
        point = map.pixel_to_world(centroids[label][0]*u.pixel, centroids[label][1]*u.pixel)
        
        for other_label in unique_labels:
            if label==other_label:
                continue
            
            other_point = map.pixel_to_world(centroids[other_label][0]*u.pixel, (centroids[other_label][1])*u.pixel)
            
            if point.separation(other_point).value < threshold:
                labels[labels==other_label]=label
                unique_labels = unique_labels[unique_labels!=other_label]

        # reducing label indexes
    for idx, label in enumerate(np.unique(labels)):
        labels[labels==label]=idx
    
    # Centroids  
    regions = measure.regionprops(labels)

    group_centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions]).astype(np.uint16)

        # adding centroid for label 0
    group_centroids = np.insert(group_centroids, 0, [1024, 1024], axis=0)
    
    return labels, group_centroids


# Utilities ##################################################


import cv2 as cv
import numpy as np



def centerDisk(image):
    _, disk = cv.threshold(image,5000,255,cv.THRESH_BINARY)
    
    disk=disk.astype(np.uint8)
    
    contours, _ = cv.findContours(disk, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Assuming the disk is the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Calculate centroid
    M = cv.moments(largest_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Calculate translation required to center the centroid
    rows, cols = disk.shape
    center_x = cols // 2
    center_y = rows // 2
    translation_x = center_x - centroid_x
    translation_y = center_y - centroid_y

    # Translate the image
    centered_image = np.roll(image, translation_x, axis=1)
    centered_image = np.roll(centered_image, translation_y, axis=0)
    
    return centered_image


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
    
    most_recent_folder = folders[0]
    
    # Get list of files in the most recent folder
    files_in_folder = [f.path for f in os.scandir(most_recent_folder) if f.is_file() and contains_string in f.name]
    
    if not files_in_folder:
        print("No files found in the most recent folder")
        return None
    
    # Sort files by modification time (most recent first)
    files_in_folder.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files_in_folder[0]