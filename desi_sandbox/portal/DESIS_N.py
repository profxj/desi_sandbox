
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv

from astropy.table import Table, Column, vstack, hstack
import astropy.units as u

from astropy.io import fits, ascii

import glob
import os

def calcsignoise(subset = False, path = "/Volumes/My Passport for Mac/andes/tiles/",
                 outfile='/Volumes/My Passport for Mac/Huge_Table.fits',
                 plot=False, verbose=False):
    all_files = glob.glob(os.path.join(path,"*/*/spectra*fits"))  # takes the each spectra file
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
    for i in range(0,6):  # to go through each file, we need a for loop, when ready change range to len of 
        spec = Table.read(sub_files[i]) # reads the i file table
        hdul = fits.open(sub_files[i])  # opens the fit data that belongs to the i sub_file and gets the information from that file

        spec_z = Table.read(sub_zfiles[i])  # reads the i file table
        hdul_z = fits.open(sub_zfiles[i])  # opens the fit data that belongs to the i sub_zfile and gets the information from that file

#        b_wave = np.array(hdul[3].data)  # Takes the third row of the hdul file
#        b_flux = np.array(hdul[4].data)  # Takes the fourth row of the hdul file
#        b_ivar = np.array(hdul[5].data)  # Takes the fifth row of the hdul file

        r_wave = np.array(hdul[8].data)  # Takes the chosen row of the hdul file
        r_flux = np.array(hdul[9].data)  # Takes the chosen row of the hdul file
        r_ivar = np.array(hdul[10].data)  # Takes the chosen row of the hdul file

#        z_wave = np.array(hdul[13].data)  # Takes the chosen row of the hdul file
#        z_flux = np.array(hdul[14].data)  # Takes the chosen row of the hdul file
#        z_ivar = np.array(hdul[15].data)  # Takes the chosen row of the hdul file


        median_array = np.array([])  # this creates an empty array to use later

        for n in range(0,len(r_flux)):  # this is to create the plot. A for loop is needed to go through each array and plot wavelength vs s/n
            rflux = r_flux[n,:]  # takes the i flux value one column at a time
            rivar = r_ivar[n,:]  # takes the i ivar value one column at a time

            r_var = 1/rivar  # inverts the ivar to get var
            r_sigma = np.sqrt(r_var)  # takes the var values and takes the sqaure root of each of the values in the array
            r_sig_noise = rflux/r_sigma # divides flux by sigma to get S/N

            if plot:
                plt.plot(r_wave, r_sig_noise)  # this plots the S/N against the wavelength

            r_SN_med = np.median(r_sig_noise)  # this takes the median S/N to the plot
            median_array = np.append(median_array, r_SN_med)  # this appends the median S/N to the empty array created earlier


        if plot:
            plt.title('Spectrum %s' %i, fontsize = 15)  #  places a title and sets font size
            plt.xlabel('Wavelength', fontsize = 15)   # places a label on the x axis and sets font size
            plt.ylabel('S/N', fontsize = 15) # places a label on the y axis and sets font size

            plt.show()

        z = Table()     # creates an empty table
        z['Z'] = spec_z['Z']   # adds this column from zbest table to the empty table we created
        z['ZWARN'] = spec_z['ZWARN']   # adds this column from zbest table to the empty table we created
        z['SPECTYPE'] = spec_z['SPECTYPE']   # adds this column from zbest table to the empty table we created

        t = Table(spec)    # takes the table spec and assigns to t
        t['S_N_r'] = median_array  # adds the median S/N of each graph and adds it to t

        table = hstack([t, z])   # combines both z and t to one table

        new_tables.append(table)  # this appends the table we made above to our empty table "new_tables"

    all_tables = vstack(new_tables)  # stacks each of the table together vertically

    # Print?
    if verbose:
        print(all_tables)

    # Write
    all_tables.write(outfile, overwrite=True)

#    t_table = Table.read('/Volumes/GoogleDrive/My Drive/Huge_Table.fits')

#    t_table

# Command line execution
if __name__ == '__main__':
    calcsignoise(subset= True, path='/Volumes/My Passport for Mac/andes/tiles/',
                 outfile='/Volumes/My Passport for Mac/Huge_Table.fits', plot=False)

idx = obs['SPECTYPE'] == 'GALAXY'  
idx_1 = idx & (obs['Z'] < 0.3) 

obs_idx = obs[idx_1]

sort = obs_idx[np.argsort(obs_idx['S_N_r'])]

sort.write('/Volumes/My Passport for Mac/OtherHuge_Table.fits', overwrite=True)

sort__table = Table.read('/Volumes/My Passport for Mac/OtherHuge_Table.fits')
