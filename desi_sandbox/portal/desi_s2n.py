""" Code calculate S/N from a DESI dataset of tiles"""
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, Column, vstack, hstack

from astropy.io import fits

import glob
import os

from IPython import embed

def calc_tile_sn(subset = False, path = "/Volumes/My Passport for Mac/andes/tiles/",
                 outfile='/Volumes/My Passport for Mac/Huge_Table.fits',
                 camera='R',
                 plot=False, verbose=False):
    """

    Args:
        subset:
        path:
        outfile:
        plot:
        verbose:

    Returns:

    """
    all_files = glob.glob(os.path.join(path,"*/*/coadd*fits"))  # takes the each spectra file
    z_files = glob.glob(os.path.join(path,"*/*/zbest*fits"))  # takes each zbest files

    if subset:
        sub_files = all_files[0:10]  # takes the first 10 files to start off
        sub_zfiles = z_files[0:10]
    else:
        sub_files = all_files
        sub_zfiles = z_files

    sub_files.sort()   # sorts each file numerically
    sub_zfiles.sort()

    new_tables = []   # creates an empty table
    for i in range(0,len(all_files)):  # to go through each file, we need a for loop
        #
        if (i % 50) == 0:
            print("File i={} of {}".format(i, len(all_files)))
        #
        hdul = fits.open(sub_files[i])  # opens the fit data that belongs to the i sub_file and gets the information from that file

        spec_z = Table.read(sub_zfiles[i])  # reads the i file table

        flux = hdul['{:s}_FLUX'.format(camera)].data  # Takes the chosen row of the hdul file
        ivar = hdul['{:s}_IVAR'.format(camera)].data  # Takes the chosen row of the hdul file

        # S/N
        S_N = np.median(flux * np.sqrt(ivar), axis=1)

        '''
        for n in range(r_flux.shape[0]):  # this is to create the plot. A for loop is needed to go through each array and plot wavelength vs s/n
            rflux = r_flux[n,:]  # takes the i flux value one column at a time
            rivar = r_ivar[n,:]  # takes the i ivar value one column at a time

            r_var = np.zeros_like(rivar)
            gdp = rivar > 0.  # Only good values
            r_var[gdp] = 1./rivar[gdp]  # inverts the ivar to get var
            r_sigma = np.sqrt(r_var)  # takes the var values and takes the sqaure root of each of the values in the array
            r_sig_noise = rflux[gdp]/r_sigma[gdp] # divides flux by sigma to get S/N

            if plot:
                plt.plot(r_wave, r_sig_noise)  # this plots the S/N against the wavelength

            r_SN_med = np.median(r_sig_noise)  # this takes the median S/N to the plot
            median_array = np.append(median_array, r_SN_med)  # this appends the median S/N to the empty array created earlier
        '''

        if plot:
            plt.title('Spectrum %s' %i, fontsize = 15)  #  places a title and sets font size
            plt.xlabel('Wavelength', fontsize = 15)   # places a label on the x axis and sets font size
            plt.ylabel('S/N', fontsize = 15) # places a label on the y axis and sets font size

            plt.show()

        # Table time
        z = Table()     # creates an empty table
        z['Z'] = spec_z['Z']   # adds this column from zbest table to the empty table we created
        z['ZWARN'] = spec_z['ZWARN']   # adds this column from zbest table to the empty table we created
        z['SPECTYPE'] = spec_z['SPECTYPE']   # adds this column from zbest table to the empty table we created

        t = Table(hdul['FIBERMAP'].data)    # takes the table spec and assigns to t
        t['S_N_{:s}'.format(camera.lower())] = S_N

        table = hstack([t, z])   # combines both z and t to one table

        new_tables.append(table)  # this appends the table we made above to our empty table "new_tables"

        # Clean up
        hdul.close()

    # Stack em all!
    all_tables = vstack(new_tables)  # stacks each of the table together vertically

    # Print?
    if verbose:
        print(all_tables)

    # Write
    all_tables.write(outfile, overwrite=True)
    print("Wrote {}".format(outfile))


# Command line execution
if __name__ == '__main__':
    # Madalyn
    if False:
        calc_tile_sn(subset= True, path='/Volumes/My Passport for Mac/andes/tiles/',
                 outfile='/Volumes/My Passport for Mac/Huge_Table.fits', plot=False)

    # JXP
    calc_tile_sn(path=os.path.join(os.getenv('DESI_ANDES'), 'tiles'),
                 outfile=os.path.join(os.getenv('DESI_UMAP'), 'andes_S2N_spec.fits'), plot=False)

    if False:
        idx = obs['SPECTYPE'] == 'GALAXY'
        idx_1 = idx & (obs['Z'] < 0.3)

        obs_idx = obs[idx_1]

        sort = obs_idx[np.argsort(obs_idx['S_N_r'])]

        sort.write('/Volumes/My Passport for Mac/OtherHuge_Table.fits', overwrite=True)

        sort__table = Table.read('/Volumes/My Passport for Mac/OtherHuge_Table.fits')
