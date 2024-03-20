
import bz2
import re
from astropy.io import fits
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames
from sunpy.map.header_helper import make_heliographic_header


def readFitsBz2(path):
    with bz2.BZ2File(path) as decompressed_file:
        with fits.open(decompressed_file) as hdul:
            data = np.flip(hdul[0].data,axis=0)
    return data


def getObstime(filename):

    filename = filename.split('/')[-1]

    pattern = r'\d{8}_\d{8}'

    match = re.search(pattern, filename)

    if match:

        date_time_str = match.group(0)
        
        formatted_datetime = f'{date_time_str[:4]}-{date_time_str[4:6]}-{date_time_str[6:8]} {date_time_str[9:11]}:{date_time_str[11:13]}:{date_time_str[13:15]}'

        return formatted_datetime
    
    else:
        return -1
    

def carrington(filename):
    
    data = readFitsBz2(filename)
    obs_time = getObstime(filename)

    coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=obs_time,
                 observer='earth', frame=frames.Helioprojective)
    
    header = sunpy.map.make_fitswcs_header(data, coord,
                                       reference_pixel=[1024, 1024]*u.pixel,
                                       scale=[1.2, 1.2]*u.arcsec/u.pixel,
                                       wavelength=6563*u.angstrom)
    
    aia_map = sunpy.map.Map(data, header)

    shape = data.shape
    carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')

    outmap = aia_map.reproject_to(carr_header)

    return outmap