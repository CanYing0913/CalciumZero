import logging
import matplotlib.pyplot as plt
import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

fnames = [
    'E:\\caiman_data\\caiman2fluro\\cropped registered [XYCTZ] cropped case1 Movie_57.tif']  # filename to be processed
fnames = [download_demo(fnames[0])]

# dataset dependent parameters
frate = 10  # movie frame rate
decay_time = 0.4  # length of a typical transient in seconds

# motion correction parameters
motion_correct = False  # flag for performing motion correction
pw_rigid = False  # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (3, 3)  # size of high pass spatial filtering, used in 1p data
max_shifts = (5, 5)  # maximum allowed rigid shift
strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'  # replicate values along the boundaries

mc_dict = {
    'fnames': fnames,
    'fr': frate,
    'decay_time': decay_time,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}

opts = params.CNMFParams(params_dict=mc_dict)


def motion_correction():
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=None, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
            plt.subplot(1, 2, 1);
            plt.imshow(mc.total_template_rig)  # % plot template
            plt.subplot(1, 2, 2);
            plt.plot(mc.shifts_rig)  # % plot rigid shifts
            plt.legend(['x shifts', 'y shifts'])
            plt.xlabel('frames')
            plt.ylabel('pixels')

        bord_px = 0 if border_nan is 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(fnames, base_name='memmap_',
                                   order='C', border_to_0=0, dview=None)


# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')

# parameters for source extraction and deconvolution
p = 1  # order of the autoregressive system
K = None  # upper bound on number of components per patch, in general None
gSig = (3, 3)  # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (13, 13)  # average diameter of a neuron, in general 4*gSig+1
Ain = None  # possibility to seed with predetermined binary masks
merge_thr = .7  # merging threshold, max correlation allowed
rf = 40  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20  # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2  # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1  # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0  # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0  # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .8  # min peak value from correlation image
min_pnr = 5  # min peak to noise ration from PNR image
ssub_B = 2  # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,  # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',  # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,  # just leave as is
                                'center_psf': True,  # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                }
                   )

# compute some summary images (correlation and peak to noise)
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0],
                                                   swap_dim=False)  # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
nb_inspect_correlation_pnr(cn_filter, pnr)

cnm = cnmf.CNMF(n_processes=6, dview=None, Ain=Ain, params=opts)
cnm.fit(images)

# %% COMPONENT EVALUATION
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier

min_SNR = 3  # adaptive way to set threshold on the transient size
r_values_min = 0.85  # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

print(' ***** ')
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))

# %% plot contour plots of accepted and rejected components
cnm.estimates.plot_contours_nb(img=cn_filter, idx=cnm.estimates.idx_components)

# accepted components
cnm.estimates.hv_view_components(img=cn_filter, idx=cnm.estimates.idx_components,
                                 denoised_color='red', cmap='gray')
# rejected components
cnm.estimates.hv_view_components(img=cn_filter, idx=cnm.estimates.idx_components_bad,
                                 denoised_color='red', cmap='gray')
myidx = cnm.estimates.idx_components
print(len(myidx))

(x,y) = cnm.estimates.A.shape
print(y)

