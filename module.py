
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

    scale = [hdu.header['NAXIS1']/(2*hdu.header['RSUN_OBS']), hdu.header['NAXIS2']/(2*hdu.header['RSUN_OBS'])]
        
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[hdu.header['CRPIX1'], hdu.header['CRPIX2']]*u.pixel,
                                        scale=scale*u.arcsec/u.pixel)
    
    return sunpy.map.Map(hdu.data, header)
    

def carrington(filename):
    
    hdu = readFitsBz2(filename)

    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=hdu.header['DATE_OBS'],
                    observer='earth', frame=frames.Helioprojective)

    scale = [hdu.header['NAXIS1']/(2*hdu.header['RSUN_OBS']), hdu.header['NAXIS2']/(2*hdu.header['RSUN_OBS'])]
        
    header = sunpy.map.make_fitswcs_header(hdu.data, coord,
                                        reference_pixel=[hdu.header['CRPIX1'], hdu.header['CRPIX2']]*u.pixel,
                                        scale=scale*u.arcsec/u.pixel)

    map = sunpy.map.Map(hdu.data, header)

    shape = hdu.data.shape
    carr_header = make_heliographic_header(map.date, map.observer_coordinate, shape, frame='carrington')

    outmap = map.reproject_to(carr_header)

    return outmap
