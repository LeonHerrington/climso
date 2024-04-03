
import bz2
from astropy.io import fits

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_heliographic_header



def readFitsBz2(path):
    decompressed_file = bz2.BZ2File(path)
    hdul =  fits.open(decompressed_file)
    return hdul[0]


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
    
    return sunpy.map.Map(hdu.data, header)


def carrington(filename, weights=None):
    
    hdu = readFitsBz2(filename)
    
    if weights.any():
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

    mu = (weights / np.nanmax(weights))

    weights = np.ones(mu.shape) - mu
    
    return weights, mu

def flatten(map):
    weights, _ = getWeights(map)
    
    flattened = map.data + 9100*weights
    
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
    cv.circle(umbra, (1024,1024), int(map.meta['rsun_obs']),color=(0,0,0), thickness = 50);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    sunspot = cv.morphologyEx(sunspot.astype(np.uint8), cv.MORPH_OPEN, kernel)
    cv.circle(sunspot, (1024,1024), int(map.meta['rsun_obs']),color=(0,0,0), thickness = 50);

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    penumbra = cv.morphologyEx(penumbra.astype(np.uint8), cv.MORPH_OPEN, kernel)
    cv.circle(penumbra, (1024,1024), int(map.meta['rsun_obs']),color=(0,0,0), thickness = 50);
    
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

def getMostRecentFile(directory):
    # Get list of folders in the directory
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    
    # Sort folders by modification time (most recent first)
    folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not folders:
        print("No folders found in the directory")
        return None
    
    most_recent_folder = folders[0]
    
    # Get list of files in the most recent folder
    files_in_folder = [f.path for f in os.scandir(most_recent_folder) if f.is_file()]
    
    if not files_in_folder:
        print("No files found in the most recent folder")
        return None
    
    # Sort files by modification time (most recent first)
    files_in_folder.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files_in_folder[0]