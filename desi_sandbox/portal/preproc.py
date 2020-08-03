""" Codes to preprocess a list of DESI spectra
Follows/adopts/adapts codes by Itamar Reis
"""
import os
import gc

import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from numba import njit, prange

from astropy.io import fits
from astropy.table import Table

from IPython import embed


@njit(parallel=True)
def remove_outliers_and_nans(s, s_, ivar):
    ####
    # Use nearby pixels to remove 3 sigma outlires and nans
    ###

    # Nan me
    s[ivar <= 0] = np.nan
    s_[ivar <= 0] = np.nan

    nof_features = s.size
    d = 5
    for f in prange(d, nof_features - d):
        val = s[f]
        leave_out = np.concatenate((s[f - d:f], s[f + 1:f + d]))
        leave_out_mean = np.nanmean(leave_out)
        leave_out_std = np.nanstd(leave_out)

        if abs(val - leave_out_mean) > 3 * leave_out_std:
            s_[f] = leave_out_mean

        d_ = d
        while not np.isfinite(s_[f]):
            val = s[f]
            d_ = d_ + 1
            leave_out = np.concatenate((s[f - d_:f], s[f + 1:f + d_]))
            leave_out_mean = np.nanmean(leave_out)
            s_[f] = leave_out_mean

    return s_


def load_spec(targ_tbl, outfile, camera='R'):
    """
    Load the spectra for a given camera and write to a Numpy file

    Args:
        targ_tbl (astropy.table.Table):
        outfile (str):
        camera (str, optional:
    """

    tileid = []  # creates an empty array
    for row in targ_tbl:
        file = str(row['PETAL_LOC']) + '-' + str(row['TILEID']) + '-' + str(row[
                                                                                'NIGHT'])  # goes through each element of the array and grabs the wanted elements of each one. This then combines them in the right format
        file_tileid = str(row['TILEID']) + '/' + str(row[
                                                         'NIGHT']) + '/coadd-' + file  # this grabs the necessary element of each array and combines them to make part of our path in the next cell
        tileid.append(file_tileid)  # appends the file created above to them empty array

    # this combines all of the elements grabbed above to make a filepath
    path = os.path.join(os.getenv('DESI_ANDES'), 'tiles')
    files = [os.path.join(path, x+'.fits') for x in tileid]

    FIBER = targ_tbl['FIBER'].data  # Takes the chosen row of the hdul file

    # Loop on the spectra
    all_flux, all_ivar, all_wave = [], [], []
    for i, ifile in enumerate(files):
        # opens the fit data that belongs to the i sub_file and gets the information from that file
        if (i % 100) == 0:
            print('i = ', i)
        hdul = fits.open(ifile)

        wave = hdul['{:s}_WAVELENGTH'.format(camera)].data  # Takes the chosen row of the hdul file
        flux = hdul['{:s}_FLUX'.format(camera)].data  # Takes the chosen row of the hdul file
        ivar = hdul['{:s}_IVAR'.format(camera)].data  # Takes the chosen row of the hdul file
        fibermap = hdul['FIBERMAP'].data  # Takes the chosen row of the hdul file

        fibers = fibermap['FIBER']

        #     print(FIBER[i])     # prints each element of FIBER
        index = np.where(np.in1d(fibers, FIBER[i]))  # grabs index is where fibers and FIBER matches
        index_ = index[0]  # converts the first element of the tuple created and converts it to a list.
        #     print(index_[0])  # prints the first element of the list

        # Save
        all_flux.append(flux[index_[0], :].astype(np.float32))  # plugs in the index found above and finds the matching spectrum
        all_ivar.append(ivar[index_[0], :].astype(np.float32))  # plugs in the index found above and finds the matching spectrum
        all_wave.append(wave.copy())

        # Cleanup
        for key in ['{:s}_WAVELENGTH'.format((camera)),
                    '{:s}_FLUX'.format((camera)),
                    '{:s}_IVAR'.format((camera)),
                    'FIBERMAP'.format((camera))]:
            del hdul[key].data
        del wave, flux, ivar, fibermap, fibers
        hdul.close()
        gc.collect()

    # Concatenate
    flux = np.concatenate(all_flux).reshape((len(targ_tbl), all_flux[0].size))
    del all_flux
    ivar = np.concatenate(all_ivar).reshape((len(targ_tbl), all_ivar[0].size))
    del all_ivar
    wave = np.concatenate(all_wave).reshape((len(targ_tbl), all_wave[0].size))
    del all_wave

    # saves the multiple arrays to one file
    np.savez_compressed(outfile, flux=flux, ivar=ivar, wave=wave, overwrite=True)
    print("Wrote: {:s}".format(outfile))

def clean_spec(raw_spec_file, cleaned_spec_file, nmedian=5):

    # Load
    specz = np.load(raw_spec_file)
    flux = specz['flux']
    ivar = specz['ivar']

    specs_final = np.zeros(flux.shape)
    for i in range(flux.shape[0]):
        if np.sum(ivar[i] <= 0) > 100:
            print('Spectrum {} is junk'.format(i))
            specs_final[i] = np.nan
            continue
        #if (i % 100 == 0):
        #    print('i = ', i, np.sum(ivar[i] <= 0))
        s = flux[i].copy()
        s_ = s.copy()
        # remove outliers and nans
        specs_final[i] = remove_outliers_and_nans(s, s_, ivar[i])
        # 5 pixel median filter (to remove some of the noise)
        specs_final[i] = medfilt(specs_final[i], nmedian)

    # saves the multiple arrays to one file
    np.savez_compressed(cleaned_spec_file, flux=flux, wave=specz['wave'], overwrite=True)
    print("Wrote: {:s}".format(cleaned_spec_file))



def rest_and_rebin(spec_file, targ_tbl):
    pass


def rebin(wavelength, flux, new_wavelength):
    """
    Resample and rebin the input Sightline object's data to a constant dlambda/lambda dispersion.

    Parameters
    ----------
    wavelength
    flux
    new_wavelength

    Returns
    -------
    new_flux

    """
    # TODO -- Add inline comments
    #c = 2.9979246e8
    #dlnlambda = np.log(1 + v / c)
    #wavelength = 10 ** sightline.loglam
    #max_wavelength = wavelength[-1]
    #min_wavelength = wavelength[0]
    #pixels_number = int(np.round(np.log(max_wavelength / min_wavelength) / dlnlambda)) + 1
    #new_wavelength = wavelength[0] * np.exp(dlnlambda * np.arange(pixels_number))

    npix = len(wavelength)
    wvh = (wavelength + np.roll(wavelength, -1)) / 2.
    wvh[npix - 1] = wavelength[npix - 1] + \
                    (wavelength[npix - 1] - wavelength[npix - 2]) / 2.
    dwv = wvh - np.roll(wvh, 1)
    dwv[0] = 2 * (wvh[0] - wavelength[0])
    med_dwv = np.median(dwv)

    cumsum = np.cumsum(flux * dwv)
    #cumvar = np.cumsum(sightline.error * dwv, dtype=np.float64)

    fcum = interp1d(wvh, cumsum, bounds_error=False)
    #fvar = interp1d(wvh, cumvar, bounds_error=False)

    nnew = len(new_wavelength)
    nwvh = (new_wavelength + np.roll(new_wavelength, -1)) / 2.
    nwvh[nnew - 1] = new_wavelength[nnew - 1] + \
                     (new_wavelength[nnew - 1] - new_wavelength[nnew - 2]) / 2.

    bwv = np.zeros(nnew + 1)
    bwv[0] = new_wavelength[0] - (new_wavelength[1] - new_wavelength[0]) / 2.
    bwv[1:] = nwvh

    newcum = fcum(bwv)
    #newvar = fvar(bwv)

    new_fx = (np.roll(newcum, -1) - newcum)[:-1]
    #new_var = (np.roll(newvar, -1) - newvar)[:-1]

    # Normalize (preserve counts and flambda)
    new_dwv = bwv - np.roll(bwv, 1)
    new_fx = new_fx / new_dwv[1:]
    # Preserve S/N (crudely)
    med_newdwv = np.median(new_dwv)
    #new_var = new_var / (med_newdwv / med_dwv) / new_dwv[1:]

    '''
    left = 0
    while np.isnan(new_fx[left]) | np.isnan(new_var[left]):
        left = left + 1
    right = len(new_fx)
    while np.isnan(new_fx[right - 1]) | np.isnan(new_var[right - 1]):
        right = right - 1

    test = np.sum((np.isnan(new_fx[left:right])) | (np.isnan(new_var[left:right])))
    assert test == 0, 'Missing value in this spectra!'

    sightline.loglam = np.log10(new_wavelength[left:right])
    sightline.flux = new_fx[left:right]
    sightline.error = new_var[left:right]
    '''

    return new_fx

# Command line execution
if __name__ == '__main__':

    # Files
    high_SN_table_file = os.path.join(os.getenv('DESI_UMAP'), 'andes_glxy_zLT03.fits')
    spec_r_file = os.path.join(os.getenv('DESI_UMAP'), 'andes_glxy_zLT03_rspec.npz')
    spec_r_clean_file = os.path.join(os.getenv('DESI_UMAP'), 'andes_glxy_zLT03_rspec_clean.npz')

    # Load galaxy table
    high_SN = Table.read(high_SN_table_file)

    # Load/write em all (not just 1000)
    if False:
        load_spec(high_SN, spec_r_file)

    # Clean
    clean_spec(spec_r_file, spec_r_clean_file)
