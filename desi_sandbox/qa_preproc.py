""" Module for assessing and generating QA on preproc routines"""

import os
import numpy as np
import glob

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.table import Table
from astropy import stats

from pypeit import ginga
from pypeit import utils

from desispec import preproc

from frb.figures import utils as ffutils

from IPython import embed

dpath = '/home/xavier/DESI/DESI_SCRATCH/minisv2/data/20200304/'
oclr = 'orange'
nclr = 'b'


def qck_bias(cameras=['z3'], debug=False):
    """
    Generate and write to disk a Bias image

    Args:
        cameras (list):

    """

    dpath = '/home/xavier/DESI/DESI_SCRATCH/minisv2/data/20200304/'
    bias_files = glob.glob(dpath+'000*/desi-*.fz')
    bias_files.sort()


    # Load em
    for camera in cameras:
        bias_imgs = None
        for ss, bias_file in enumerate(bias_files):
            print("Loading {}".format(bias_file))
            hdul = fits.open(bias_file)
            #- Check if this file uses amp names 1,2,3,4 (old) or A,B,C,D (new)
            hdu = hdul[camera.upper()]
            header = hdu.header
            if bias_imgs is None:
                bias_imgs = np.zeros((len(bias_files), hdu.data.shape[0], hdu.data.shape[1]), dtype=float)
            bias_imgs[ss,:,:] = hdu.data.astype(float)
        # Stack em
        bias_img, _, _ = stats.sigma_clipped_stats(bias_imgs, axis=0)
        if debug:
            ginga.show_image(bias_img)
            embed(header='51 of qa_preproc')

        # Write
        hdu0 = fits.PrimaryHDU(bias_img)
        hdul = fits.HDUList([hdu0])
        outfile = os.path.join(dpath, 'bias_{}.fits'.format(camera))
        hdul.writeto(outfile, overwrite=True)

        print("Wrote: {}".format(outfile))


def bias_stats(camera='z3', use_overscan_row=True):
    """
    Measure stats on bias-subtracted frames

    Args:
        camera:

    Returns:

    """
    dpath = '/home/xavier/DESI/DESI_SCRATCH/minisv2/data/20200304/'
    bias_files = glob.glob(dpath+'000*/desi-*.fz')
    bias_files.sort()

    # Load bias images
    hdul = fits.open(bias_files[0])
    hdu = hdul[camera.upper()]
    smnum = hdu.header['SPECID']
    new_bias_file = os.path.join(dpath, 'bias_{}.fits'.format(camera))
    new_bias = fits.open(new_bias_file)[0].data
    old_bias_file = os.path.join(os.getenv(
        'DESI_SPECTRO_CALIB'), 'ccd', 'bias-sm{}-{}-20191021.fits'.format(smnum, camera[0]))
    old_bias = fits.open(old_bias_file)[0].data

    calib_file = os.path.join(os.getenv('DESI_SPECTRO_CALIB'), 'spec', 'sm{}'.format(smnum),
                              'sm{}-{}.yaml'.format(smnum, camera[0]))

    # Loop and process
    stat_tbl = None
    for ss, bias_file in enumerate(bias_files):
        hdul = fits.open(bias_file)
        # Process
        tdict = dict(file=[os.path.basename(bias_file)])
        lbias = 'new'
        for lbias, bimg in zip(['new', 'old'], [new_bias, old_bias]):
            img = preproc.preproc(hdu.data, hdu.header, hdul[0].header, pixflat=False, mask=False,
                                  nocrosstalk=True, ccd_calibration_filename=calib_file,
                                  use_overscan_row=use_overscan_row,
                                  dark=False, flag_savgol=False, bias_img=bimg)
            # Stats
            amp_ids = preproc.get_amp_ids(hdu.header)
            for amp in amp_ids:
                kk = preproc.parse_sec_keyword(hdu.header['CCDSEC' + amp])
                zero, rms = preproc._overscan(img.pix[kk])
                tdict[lbias+amp+'_zero'] = [zero]
                tdict[lbias+amp+'_rms'] = [rms]
        # Add to Table
        if stat_tbl is None:
            stat_tbl = Table(tdict)
        else:
            stat_tbl.add_row(tdict)
    # Write
    if use_overscan_row:
        root = '_'
    else:
        root = '_norow_'
    outbase = 'bias_stats{}{}.fits'.format(root,camera)
    outfile = os.path.join(dpath, 'Stats', outbase)
    stat_tbl.write(outfile, overwrite=True)
    print("Wrote: {}".format(outfile))

def fig_bias_stats(camera='z3', use_overscan_row=True):

    if use_overscan_row:
        root = '_'
    else:
        root = '_norow_'
    outfile = 'fig_bias_stats{}{}.png'.format(root, camera)

    # Load table
    base = 'bias_stats{}{}.fits'.format(root,camera)
    tblfile = os.path.join(dpath, 'Stats', base)
    stat_tbl = Table.read(tblfile)
    nexp = len(stat_tbl)

    plt.figure(figsize=(10, 6))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)

    clrs = ['k', 'b', 'r', 'g']

    for ss, lbias in enumerate(['old', 'new']):
        ax = plt.subplot(gs[ss])

        for tt, amp in enumerate(['A','B','C','D']):
            xval = np.arange(nexp)+1 - 0.2 + tt*0.1
            yval = stat_tbl[lbias+amp+'_zero'].data
            sig = stat_tbl[lbias+amp+'_rms'].data
            lbl = amp
            ax.scatter(xval, yval, color=clrs[tt], marker='s', label=lbl)
            #ax.errorbar(xval, yval, sig, color=clrs[tt], fmt='s', label=lbl, capsize=10)
        ax.set_xlim(0, nexp+1)
        ax.set_ylim(-0.2, 0.2)

        # Label
        ax.text(0.05, 0.90, lbias, transform=ax.transAxes, fontsize=23, ha='left', color='black')
        ffutils.set_fontsize(ax, 13.)

    legend = ax.legend(loc='upper right', scatterpoints=1, borderpad=0.2, fontsize=13)

    # Layout and save
    print('Writing {:s}'.format(outfile))
    plt.tight_layout(pad=0.2, h_pad=0., w_pad=0.1)
    # plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(dpath, 'Figures', outfile), dpi=500)  # , bbox_inches='tight')
    plt.close()

def grab_stack_bias_img(camera, old=True):
    dpath = '/home/xavier/DESI/DESI_SCRATCH/minisv2/data/20200304/'
    if old:
        bias_files = glob.glob(dpath + '000*/desi-*.fz')
        bias_files.sort()
        hdu = fits.open(bias_files[0])[camera.upper()]
        smnum = hdu.header['SPECID']
        bias_file = os.path.join(os.getenv(
            'DESI_SPECTRO_CALIB'), 'ccd', 'bias-sm{}-{}-20191021.fits'.format(smnum, camera[0]))
    else:
        bias_file = os.path.join(dpath, 'bias_{}.fits'.format(camera))
        hdu = None
    bias = fits.open(bias_file)[0].data
    return bias, hdu

def visual_inspect(camera='b3', idx=0, old=True):
    """
    View an image bias-subtracted by the old or new fig

    Args:
        camera:
        idx:
        old:

    Returns:

    """
    dpath = '/home/xavier/DESI/DESI_SCRATCH/minisv2/data/20200304/'
    bias_files = glob.glob(dpath+'000*/desi-*.fz')
    bias_files.sort()
    bias_file = bias_files[idx]
    print("Inspecting: {}".format(bias_file))

    # Load
    hdul = fits.open(bias_file)
    hdu = hdul[camera.upper()]
    bias = hdu.data

    stack_bias, _ = grab_stack_bias_img(camera, old=old)

    # View
    ginga.show_image(bias-stack_bias)

def compare_vertical(camera='z3', AMP='D'):

    outfile = 'fig_bias_compare_{}-{}.png'.format(camera, AMP)

    old_bias, hdu = grab_stack_bias_img(camera, old=True)
    new_bias, _ = grab_stack_bias_img(camera, old=False)

    jj = preproc.parse_sec_keyword(hdu.header['DATASEC' + AMP])
    #kk = preproc.parse_sec_keyword(hdu.header['' + AMP])
    ov_col = preproc.parse_sec_keyword(hdu.header['BIASSEC' + AMP])
    old_sub_img = old_bias[jj]
    new_sub_img = new_bias[jj]
    med_new = np.median(new_bias[ov_col])

    # Smash and plot
    old_smash = np.median(old_sub_img, axis=0)
    new_smash = np.median(new_sub_img-med_new, axis=0)

    plt.clf()
    ax = plt.gca()
    ax.plot(old_smash, label='old', color=oclr)
    ax.plot(new_smash, label='new', color=nclr)

    legend = ax.legend(loc='upper right', scatterpoints=1, borderpad=0.2, fontsize=13)
    ax.set_xlabel('Col')
    ax.set_ylabel('Median Stacked ADU')
    plt.show()
    embed()

    # Layout and save
    print('Writing {:s}'.format(outfile))
    plt.tight_layout(pad=0.2, h_pad=0., w_pad=0.1)
    # plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(dpath, 'Figures', outfile), dpi=500)  # , bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Generate bias frames
    cameras=['b3', 'r3', 'z3', 'b6', 'r6', 'z6']

    # Bias images
    #qck_bias(cameras=cameras)
    #qck_bias(cameras=['z3'])

    # Stat figures
    if True:
        for camera in cameras:
            bias_stats(camera=camera, use_overscan_row=False)
            fig_bias_stats(camera=camera, use_overscan_row=False)

    # Inspect
    #visual_inspect(camera='z6', old=False)
    compare_vertical(camera='z3', AMP='D')

