
import bz2
from astropy.io import fits
import numpy as np

def readFits(path):
    with bz2.BZ2File(path) as decompressed_file:
        with fits.open(decompressed_file) as hdul:
            data = np.flip(hdul[0].data,axis=0)
    return data