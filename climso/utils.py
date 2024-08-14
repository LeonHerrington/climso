import numpy as np
import cv2 as cv
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import sunpy.map
from sunpy.coordinates import frames


### FITS reading ***************************************************************************************

def getHeader(hdu:fits.PrimaryHDU) -> sunpy.util.MetaDict:
    """
    Parameters
    ------
    hdu (astropy.io.fits.PrimaryHDU) : HDU from FITS file.
    
    Returns
    ------
    header (sunpy.util.MetaDict) : The header information required for making a sunpy.map.GenericMap.
    """
    
    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                 observer='earth', frame=frames.Helioprojective)
    
    scale  = [0.5*(hdu.header['NAXIS1']-100)/hdu.header['RSUN_OBS'], 0.5*(hdu.header['NAXIS2']-100)/hdu.header['RSUN_OBS']]
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[1024, 1024]*u.pixel,
                                        scale=scale*u.arcsec/u.pixel)
    
    header['rsun_obs'] = hdu.header['rsun_obs']
    
    return header


def toSunpyMap(filename, center_disk=False) -> sunpy.map.GenericMap:
    """
    Parameters
    ------
    filename (str) : Fits filename.
    
    center_disk (bool) : If true centers the disk.
    
    Returns
    ------
    sunpy.map.GenericMap
    """

    with fits.open(filename) as hdul:

        if center_disk: hdul[0] = centerDisk(hdul[0])
        
        header = getHeader(hdul[0])
        
        hdul[0].data = np.flip(hdul[0].data, axis=0)
                
        map = sunpy.map.Map(hdul[0].data, header, map_type='generic_map')
    
    return map


### Image Pre-Processing *******************************************************************************

def centerDisk(hdu):
    """
    Parameters
    ------
    hdu (astropy.io.fits.PrimaryHDU) : HDU from FITS file.
    
    Returns
    ------
    centered disk hdu (astropy.io.fits.PrimaryHDU)
    """
    
    # The threshold (5000) may need to be adapted if there is a change with the images.
    _, disk = cv.threshold(hdu.data, 5000, 255, cv.THRESH_BINARY) 
    
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

def medianFlatten(map) -> np.ndarray[np.dtype[np.uint16]]:
    """
    Parameters
    ------
    map (sunpy.map.GenericMap)
    
    Returns
    ------
    Flattened image (ndarray[dtype[uint16], Any])
    """
    # Convert to uint8
    image_uint16 = map.data.copy()

    min_val = image_uint16.min()
    max_val = image_uint16.max()

    image_uint8 = ((image_uint16 - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # median blur and convert back to uint16
    medblur_uint8 = cv.medianBlur(image_uint8, 255)
    medblur_uint16 = ((medblur_uint8.astype(np.float32) / 255 * max_val) + min_val).astype(np.uint16)
    
    # invert
    inverted = np.max(medblur_uint16)-medblur_uint16
    
    flattened = image_uint16 + 1.0*inverted
    
    return flattened


### Misc ***********************************************************************************************

def get_mu(map):
    
    coordinates = sunpy.map.all_coordinates_from_map(map)
    weights     = coordinates.transform_to("heliocentric").z.value
    mu = np.array(weights / np.nanmax(weights))
    
    return  mu