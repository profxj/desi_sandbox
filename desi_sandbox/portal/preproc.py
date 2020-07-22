""" Codes to preprocess a list of DESI spectra
Follows/adopts/adapts codes by Itamar Reis
"""

import numpy
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
        leave_out = numpy.concatenate((s[f - d:f], s[f + 1:f + d]))
        leave_out_mean = numpy.nanmean(leave_out)
        leave_out_std = numpy.nanstd(leave_out)

        if abs(val - leave_out_mean) > 3 * leave_out_std:
            s_[f] = leave_out_mean

        d_ = d
        while not numpy.isfinite(s_[f]):
            val = s[f]
            d_ = d_ + 1
            leave_out = numpy.concatenate((s[f - d_:f], s[f + 1:f + d_]))
            leave_out_mean = numpy.nanmean(leave_out)
            s_[f] = leave_out_mean

    return s_

def load_spec(targ_tbl, path="/Volumes/My Passport for Mac/andes/tiles/"):

    # Loop on the spectra

    # Splice
    #  Take all of the b camera
    #  Take all of the z camera

