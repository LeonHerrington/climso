
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
    
    
def toSunpyMap(filename):
    hdu = readFitsBz2(filename)

    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                 observer='earth', frame=frames.Helioprojective)
        
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[hdu.header['CRPIX1'], hdu.header['CRPIX2']]*u.pixel,
                                        scale=[1.2, 1.2]*u.arcsec/u.pixel)
    
    return sunpy.map.Map(hdu.data, header)
    

def carrington(filename, weights=None):
    
    hdu = readFitsBz2(filename)
    
    if weights.any():
        hdu.data = hdu.data * weights

    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                    observer='earth', frame=frames.Helioprojective)
        
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[hdu.header['CRPIX1'], hdu.header['CRPIX2']]*u.pixel,
                                        scale=[1.2, 1.2]*u.arcsec/u.pixel)

    map = sunpy.map.Map(hdu.data, header)

    carr_header = make_heliographic_header(map.date, map.observer_coordinate, hdu.data.shape, frame='carrington')

    outmap = map.reproject_to(carr_header)

    return outmap