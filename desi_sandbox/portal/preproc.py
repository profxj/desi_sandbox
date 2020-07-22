""" Codes to preprocess a list of DESI spectra
Follows/adopts/adapts codes by Itamar Reis
"""

import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from numba import njit
import tqdm


@njit
def remove_outliers_and_nans(s, s_):
    ####

    # Use nearby pixels to remove 3 sigma outlires and nans

    ###
    nof_features = s.size
    d = 5
    for f in range(d, nof_features - d):
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

def load_spec(targ_tbl, path="/Volumes/My Passport for Mac/andes/tiles/"):

    # Loop on the spectra
    pass

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

