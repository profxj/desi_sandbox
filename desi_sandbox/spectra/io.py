""" I/O for spectra in DESI"""
import os
import numpy as np

from astropy.io import fits

from IPython import embed

def load_andes_obj(obj_dict, camera='r'):
    """
    Load up the spectrum for an andes object from
    PETAL_LOC, TILEID, NIGHT, TILEID and FIBER

    Args:
        obj_dict (dict-like object):
        camera (str, optional):

    Returns:
        tuple: flux, ivar, wave

    """

    # goes through each element of the array and grabs the wanted elements of each one. This then combines them in the right format
    ifile = str(obj_dict['PETAL_LOC']) + '-' + str(obj_dict['TILEID']) + '-' + str(obj_dict['NIGHT'])
    # this grabs the necessary element of each array and combines them to make part of our path in the next cell
    file_tileid = str(obj_dict['TILEID']) + '/' + str(obj_dict['NIGHT']) + '/coadd-'+ifile
    if os.getenv('DESI_ANDES') is not None:  # For Madalyn
        path = os.path.join(os.getenv('DESI_ANDES'), 'tiles')
    else:
        path = '/Volumes/GoogleDrive/My Drive/andes (1)/'
    filename = os.path.join(path, file_tileid+'.fits')

    hdul = fits.open(filename)

    wave = hdul['{:s}_WAVELENGTH'.format(camera)].data  # Takes the chosen row of the hdul file
    flux = hdul['{:s}_FLUX'.format(camera)].data  # Takes the chosen row of the hdul file
    ivar = hdul['{:s}_IVAR'.format(camera)].data  # Takes the chosen row of the hdul file

    fibermap = hdul['FIBERMAP'].data  # Takes the chosen row of the hdul file

    fibers = fibermap['FIBER']
    index = np.where(np.in1d(fibers, obj_dict['FIBER']))[0][0]  # grabs index is where fibers and FIBER matches

    return flux[index, :].astype(np.float32), ivar[index, :].astype(np.float32),  wave
