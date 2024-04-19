"""
Source file for pipeline in general, OOP workflow
Last edited on Dec.31 2022
Copyright Yian Wang (canying0913@gmail.com) - 2022
"""
import argparse
from multiprocessing import Pool
import imagej
from time import perf_counter
from src.src_caiman import *
# Retrieve source
from src.src_detection import *
from src.src_stabilizer import print_param, run_plugin
from src.src_peak_caller import PeakCaller


def remove_suffix(input_string, suffix):
    if suffix and str(input_string).endswith(suffix):
        return str(input_string)[:-len(suffix)]
    return str(input_string)


def parse():
    """
    Parse function for argparse
    """
    # Set up description
    desp = 'Automated pipeline for CaImAn processing.\nIf you have multiple inputs, please place them within a ' \
           'single folder without any other files.'
    parser = argparse.ArgumentParser(description=desp)
    # Set up arguments
    # Control parameters
    parser.add_argument('-no_log', default=False, action='store_true',
                        help='Specified if do not want to have terminal printouts saved to a separate file.')
    parser.add_argument('-ijp', '--imagej-path', type=str, metavar='ImageJ-Path', required=True,
                        help='Path to local Fiji ImageJ fiji folder.')
    parser.add_argument('-wd', '--work-dir', type=str, metavar='Work-Dir', required=True,
                        help='Path to a working folder where all intermediate and overall results are stored. If not '
                             'exist then will create one automatically.')
    parser.add_argument('-input', type=str, metavar='INPUT', required=True,
                        help='Path to input/inputs folder. If you have multiple inputs file, please place them inside '
                             'a single folder. If you only have one input, either provide direct path or the path to'
                             ' the folder containing the input(without any other files!)')
    parser.add_argument('-do_s0', default=True, action="store_false", required=False, help='Skip cropping if specified')
    parser.add_argument('-do_s1', default=True, action="store_false", required=False,
                        help='Skip Stabilizer if specified.')
    parser.add_argument('-do_s2', default=True, action="store_false", required=False,
                        help='Skip CaImAn if specified.')
    # Functional parameters
    parser.add_argument('-margin', default=200, type=int, metavar='Margin', required=False,
                        help='Margin in terms of pixels for auto-cropping. Default to be 200.')
    parser.add_argument('-ij_trans', default=0, type=int, required=False,
                        help='ImageJ stabilizer parameter - Transformation. You have to specify -ij_param to use it. '
                             'Default to translation, set it to 1 if want to set it to affine.')
    parser.add_argument('-ij_maxpl', default=1, type=float, required=False,
                        help='ImageJ stabilizer parameter - MAX_Pyramid_level. You have to specify -ij_param to use '
                             'it. Default to be 1.0.')
    parser.add_argument('-ij_upco', default=0.90, type=float, required=False,
                        help='ImageJ stabilizer parameter - update_coefficient. You have to specify -ij_param to use '
                             'it. Default to 0.90.')
    parser.add_argument('-ij_maxiter', default=200, type=int, required=False,
                        help='ImageJ stabilizer parameter - MAX_iteration. You have to specify -ij_param to use it. '
                             'Default to 200.')
    parser.add_argument('-ij_errtol', default=1E-7, type=float, required=False,
                        help='ImageJ stabilizer parameter - error_rolerance. You have to specify -ij_param to use '
                             'it. Default to 1E-7.')
    parser.add_argument('-clog', default=False, action='store_true',
                        help='True if enable logging for caiman part. Default to be false.')
    parser.add_argument('-csave', default=False, action='store_true',
                        help='True if want to save denoised movie. Default to be false.')
    # Parse the arguments
    arguments = parser.parse_args()
    # Post-process arguments
    # ImageJ path and param
    # arguments.imagej_path = Path(arguments.imagej_path)
    if not os.path.exists(arguments.imagej_path):
        if not arguments.skip_1:
            raise OSError(f"[ERROR]: ImageJ path does not exist: {arguments.imagej_path}")
    arguments.ij_params = [
        arguments.ij_trans,
        arguments.ij_maxpl,
        arguments.ij_upco,
        arguments.ij_maxiter,
        arguments.ij_errtol
    ]
    # work_dir path
    # arguments.work_dir = Path(arguments.work_dir)
    if not os.path.exists(arguments.work_dir):
        print(f"Working directory {arguments.work_dir} does not exist. Attempting to create one.")
        try:
            Path(arguments.work_dir).mkdir(parents=True, exist_ok=False)
        except OSError:
            print(f"[ERROR]: OSError detected. Please check if disk exists or privilege level satisfies.")
            exit(1)
    # input folder path
    arguments.input = str(arguments.input)  # To suppress IDE warning
    if os.path.exists(arguments.input):
        if os.path.isdir(arguments.input):
            # Path to a folder of multiple inputs.
            arguments.input_root = arguments.input
            arguments.input = [f for f in os.listdir(arguments.input_root) if f[-4:] == '.tif']
        else:
            # Path to a single input file.
            temp = os.path.basename(arguments.input)
            if temp[-4:] != ".tif":
                raise FileNotFoundError(f"The input file {arguments.input} is not a tiff file.")
            arguments.input_root = remove_suffix(arguments.input, temp)
            arguments.input = [temp]
    else:
        raise FileNotFoundError(f"[ERROR]: Input file path {arguments.input} does not exist.")
    return arguments


class Pipeline(object):
    def __init__(self, queue=None, queue_id=0, log_queue=None):
        # GUI-related
        self.queue = queue
        self.queue_id = queue_id
        self.log_queue = log_queue
        self.logger = None
        self.params_dict = {
            'run': [True, True, True],
            'crop': {'margin': 200},
            'stabilizer': {
                'Transformation': 'Translation',
                'MAX_Pyramid_level': 1.0,
                'update_coefficient': 0.90,
                'MAX_iteration': 200,
                'error_tolerance': 1E-7
            },
            'caiman': {
                "mc_dict": {
                    'fnames': [""],
                    'fr': frate,
                    'decay_time': decay_time,
                    'pw_rigid': pw_rigid,
                    'max_shifts': max_shifts,
                    'gSig_filt': gSig_filt,
                    'strides': strides,
                    'overlaps': overlaps,
                    'max_deviation_rigid': max_deviation_rigid,
                    'border_nan': border_nan
                },
                "params_dict": {'method_init': 'corr_pnr',  # use this for 1 photon
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
                                'update_background_components': True,
                                # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,  # just leave as is
                                'center_psf': True,  # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                }
            }
        }
        self.is_running = False
        self.is_finished = False
        # Control sequence
        self.skip_0 = self.skip_1 = False
        self.work_dir = ''
        self.log = None
        self.process = 2
        self.cache = Path(__file__).parent.parent.joinpath('cache')
        self.cache.mkdir(exist_ok=True, parents=False)
        # Segmentation and cropping related variables
        self.do_s0 = False
        self.input_root = ''
        self.input_list = []
        self.imm1_list = []  # Intermediate result list 1, relative path
        self.done_s0 = False
        self.QCimage_s0_raw = None
        self.QCimage_s0 = None
        # ImageJ stabilizer related variables
        self.do_s1 = False
        self.ijp = Path(__file__).parent.parent.joinpath('Fiji.app')
        # self.ij = imagej.init(str(self.ijp), mode='interactive')
        self.s1_params = [
            'Translation',
            '1.0',
            '0.90',
            '200',
            '1E-7'
        ]
        self.s1_root = ''
        self.imm2_list = []  # Intermediate result list 2, relative path
        self.done_s1 = False
        self.QCimage_s1_raw = None
        self.QCimage_s1 = None
        # CaImAn related variables
        self.do_s2 = False
        self.caiman_obj = None
        self.clog = True
        self.csave = False
        self.s2_root = ''
        self.done_s2 = False
        self.outpath_s2 = ''
        self.cmobj_path = ''
        self.QCimage_s2 = None
        # Peak Caller related
        self.do_s3 = False
        self.pc_obj = []

    def log_print(self, txt: str):
        if self.log_queue:
            self.log_queue.put(txt)
        elif self.logger:
            self.logger.debug(txt)
        else:
            print(txt)
            if self.log is not None:
                self.log.write(txt + '\n')

    def parse(self):
        # Retrieve calling parameters
        arguments = parse()

        # Must only specify one skip
        assert self.skip_0 is False or self.skip_1 is False, "Duplicate skip param specified."
        self.s1_root = self.s2_root = self.work_dir
        # Use parameters to set up pipeline global info
        # Control related
        self.work_dir = arguments.work_dir
        self.skip_0 = arguments.skip_0
        self.skip_1 = arguments.skip_1
        if not arguments.no_log:
            log_path = os.path.join(self.work_dir, 'log.txt')
            self.log = open(log_path, 'w')
            self.log_print(f"log file is stored @ {log_path}")
        # Segmentation and cropping related variables
        self.input_root = arguments.input_root
        self.input_list = arguments.input
        self.margin = arguments.margin
        # ImageJ related
        if not self.skip_1:
            self.ij = imagej.init(arguments.imagej_path, mode='headless')
            self.ijp = arguments.imagej_path
            self.log_print(f"ImageJ initialized with version {self.ij.getVersion()}.")
            self.s1_params = arguments.ij_params
            print_param(self.s1_params, self.log_print)
        # CaImAn related variables
        # TODO: add caiman parameters
        self.clog = arguments.clog
        self.csave = arguments.csave
        # TODO: add peak_caller parameters
        pass
        # Get control params to determine dest list
        # TODO: need extra care for caiman mmap generation

        # End of parser. Start of post-parse processing.
        if self.skip_0:
            self.s1_root = self.input_root
            self.imm1_list = self.input_list
        elif self.skip_1:
            self.s2_root = self.input_root
            self.imm2_list = self.input_list
        return None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.log_print(f'the requested key {key} does not exist.')
                continue
            if key == 'input_list':
                if len(value) == 1 and 'cmn_obj' in value[0]:
                    with open(str(Path(self.input_root).joinpath(value[0])), 'rb') as f:
                        self.caiman_obj = pickle.load(f)
                        continue

            # Setup extra params other than s0-s2
            if key == 'params_dict':
                self.input_list = [value['input_path']]
                self.work_dir = value['output_path']
                self.do_s0, self.do_s1, self.do_s2 = value['run']
                self.params_dict = value
                continue
            if key in self.params_dict['caiman'].keys():
                self.params_dict['caiman'][key] = value
                continue
            if key in self.params_dict['caiman']['mc_dict'].keys():
                self.params_dict['caiman']['mc_dict'][key] = value
                continue
            if key in self.params_dict['caiman']['params_dict'].keys():
                self.params_dict['caiman']['params_dict'][key] = value
                continue
            setattr(self, key, value)

    def ready(self):
        if not self.input_list:
            return False, 'Input not specified'
        if self.work_dir == '':
            return False, 'Output folder not set'
        if self.do_s0:
            for input_file in self.input_list:
                if '.tif' not in input_file:
                    return False, f'Wrong input format for crop: {input_file}'
        if self.do_s1:
            if self.ijp == '':
                return False, 'ImageJ not ready'
            for input_file in self.input_list:
                if '.tif' not in input_file:
                    return False, f'Wrong input format for Stabilizer: {input_file}'
        if self.do_s2:
            pass
        return True, ''

    def s0(self):
        """
        Function to run segmentation, detection and cropping.

        Parameters:

        """

        def ps0(text: str):
            self.log_print(f"***[S0 - Detection]: {text}")

        start_time = perf_counter()
        # Segmentation and cropping
        # Scanning for bounding box for multiple input
        with Pool(processes=self.process) as pool:
            fnames = [str(Path(self.input_root).joinpath(fname)) for fname in self.input_list]
            results = pool.map(scan, fnames)
        x1, y1, x2, y2 = reduce_bbs(results)
        for idx, fname in enumerate(fnames):
            fpath_seg = Path(self.work_dir).joinpath(Path(Path(fname).stem + "_seg.tif"))
            tifffile.imwrite(fpath_seg, results[idx][-1])
        finalxcntss = [result[4] for result in results]

        # Apply the uniform bb one-by-one to each input image
        for idx, fname_i in enumerate(self.input_list):
            image_i = tifffile.imread(str(Path(self.input_root).joinpath(fname_i)))
            finalxcnts = finalxcntss[idx]
            image_crop_o = apply_bb_3d(image_i, (x1, y1, x2, y2), self.params_dict['crop']['threshold'], finalxcnts)
            # Handle output path
            fname_crop_root = remove_suffix(fname_i, '.tif') + '_crop.tif'
            fname_crop_o = os.path.join(self.work_dir, fname_crop_root)
            ps0(f"Using paths: {fname_crop_o} to save cropped result.")
            # Save imm1 data to files
            tifffile.imwrite(fname_crop_o, image_crop_o)
            self.imm1_list.append(fname_crop_root)
        self.done_s0 = True
        end_time = perf_counter()
        ps0(f"Detection finished in {int(end_time - start_time)} s.")

    def s1(self):
        def ps1(text: str):
            self.log_print(f"***[S1 - ImageJ stabilizer]: {text}")

        # ImageJ Stabilizer
        ps1(f"Stabilizer Starting.")
        results = []
        start_time = perf_counter()
        idx = 0
        while idx < len(self.imm1_list):
            imm1_list = [self.imm1_list[idx + i] for i in range(self.process) if idx + i < len(self.imm1_list)]
            idx += self.process
            with Pool(processes=len(imm1_list)) as pool:
                results = pool.starmap(run_plugin,
                                       [(
                                           self.ijp, str(Path(self.s1_root).joinpath(imm1)), self.work_dir,
                                           self.params_dict['stabilizer'])
                                           for imm1 in imm1_list])
        end_time = perf_counter()
        ps1(f"Stabilizer finished in {int(end_time - start_time)} s.")
        self.imm2_list = results  # note here is absolute path list

    def s2(self):
        def ps2(txt: str):
            self.log_print(f"***[S2 - caiman]: {txt}")

        self.cmobj_path = os.path.join(self.work_dir, "cmn_obj.cmobj")

        start_time = perf_counter()
        if self.clog:
            ps2(f"caiman logging enabled.")
            logging.basicConfig(
                format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                level=logging.DEBUG)
        fnames = [str(Path(self.input_root).joinpath(fname)) for fname in self.imm2_list]
        self.outpath_s2 = [str(Path(self.work_dir).joinpath(remove_suffix(f, '.tif') + '_caiman.tif')) for f in fnames]
        opts = params.CNMFParams(params_dict=self.params_dict['caiman']['mc_dict'])
        # Motion Correction
        if motion_correct:
            # do motion correction rigid
            mc = MotionCorrect(fnames, dview=None, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
            if pw_rigid:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                             np.max(np.abs(mc.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(mc.total_template_rig)  # % plot template
                plt.subplot(1, 2, 2)
                plt.plot(mc.shifts_rig)  # % plot rigid shifts
                plt.legend(['x shifts', 'y shifts'])
                plt.xlabel('frames')
                plt.ylabel('pixels')
                plt.show()

            bord_px = 0 if border_nan == 'copy' else bord_px
            fname_mmap = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            bord_px = 0
            fname_mmap = cm.save_memmap(fnames, base_name='memmap_', order='C', border_to_0=0, dview=None)
        ps2(f"mmap file saved to {fname_mmap}")

        # load memory mappable file
        Yr, dims, T = cm.load_memmap(fname_mmap)
        images = Yr.T.reshape((T,) + dims, order='F')

        opts.change_params(self.params_dict['caiman']['params_dict'])
        # Inspect summary images and set parameters
        # compute some summary images (correlation and peak to noise)
        cn_filter, pnr = cm.summary_images.correlation_pnr(images[::10], gSig=gSig[0],
                                                           swap_dim=False)  # change swap dim if output looks weird, it is a problem with tiffile
        # inspect the summary images and set the parameters
        # nb_inspect_correlation_pnr(cn_filter, pnr)

        # Run the CNMF-E algorithm
        start_time_cnmf = perf_counter()
        cnm = cnmf.CNMF(n_processes=8, dview=None, Ain=Ain, params=opts)
        cnm.fit(images)
        end_time_cnmf = perf_counter()
        exec_time_cnmf = end_time_cnmf - start_time_cnmf
        ps2(f"cnmf takes {exec_time_cnmf // 60}m, {int(exec_time_cnmf % 60)}s to complete.")
        # ## Component Evaluation
        # the components are evaluated in three ways:
        #   a) the shape of each component must be correlated with the data
        #   b) a minimum peak SNR is required over the length of a transient
        #   c) each shape passes a CNN based classifier
        min_SNR = 3  # adaptive way to set threshold on the transient size
        r_values_min = 0.85  # threshold on space consistency (if you lower more components will be accepted, potentially
        # with worst quality)
        cnm.params.set('quality', {'min_SNR': min_SNR,
                                   'rval_thr': r_values_min,
                                   'use_cnn': False})
        cnm.estimates.evaluate_components(images, cnm.params, dview=None)

        ps2(' ***** ')
        ps2(f'Number of total components:  {len(cnm.estimates.C)}')
        ps2(f'Number of accepted components: {len(cnm.estimates.idx_components)}')

        # Get alll detected spatial components
        x, y = cnm.estimates.A.shape
        # the index of accepted components
        myidx = cnm.estimates.idx_components

        coordinate1 = np.reshape(cnm.estimates.A[:, myidx[1]].toarray(), dims, order='F')
        bl = coordinate1 > 0

        # setup blank merge arrays. One is from merge, the other is from overlapped areas
        merged = np.where(bl is True, 0, coordinate1)
        mhits = np.where(bl is True, 0, coordinate1)
        blm = merged > 0

        for i in myidx:
            coordinate2 = np.reshape(cnm.estimates.A[:, i].toarray(), dims, order='F')
            # %% generate boolean indexing
            bl2 = coordinate2 > 0
            ct2 = np.sum(bl2)
            blm = merged > 0
            # identify the overlapped components
            bli = np.logical_and(bl2, blm)
            cti = np.sum(bli)
            # calculate the portion of the overlapped
            percent = cti / ct2
            # print(percent)
            if percent < 0.25:
                # change the label of this component
                merged = np.where(bl2 is True, i + 1, merged)
                # exclude the overlapped areas
                merged = np.where(bli is True, 0, merged)
            else:
                # put the overlapped areas here
                mhits = np.where(bli is True, 999 + i, mhits)

        np.savetxt(os.path.join(self.work_dir, "coor_merged.csv"), merged, delimiter=",")
        np.savetxt(os.path.join(self.work_dir, "coor_mhits.csv"), mhits, delimiter=",")

        # Extract DF/F values
        (components, frames) = cnm.estimates.C.shape
        ps2(f"frames: {frames}")
        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=frames)
        setattr(cnm, 'dims', dims)
        # Save input files to *.cmobj file
        setattr(cnm, 'input_files', [tifffile.imread(fname) for fname in fnames])
        self.caiman_obj = cnm
        # reconstruct denoised movie
        if self.csave:
            denoised = cm.movie(cnm.estimates.A.dot(cnm.estimates.C)).reshape(dims + (-1,), order='F').transpose(
                [2, 0, 1])
            denoised.save(self.outpath_s2)
            ps2(f"caiman denoised movie saved to {self.outpath_s2}")
        with open(self.cmobj_path, "wb") as f:
            pickle.dump(cnm, f)
            ps2(f"object cnm dumped to {self.cmobj_path}.")
        end_time = perf_counter()
        exec_time = end_time - start_time
        ps2(f"caiman finished in {exec_time // 60}m, {int(exec_time % 60)} s.")

    def s3(self):
        # peak_caller
        # slice_num = _
        if not self.do_s2:
            with open(Path(self.input_root).joinpath(self.input_list[0]), 'rb') as f:
                self.caiman_obj = pickle.load(f)
        data = self.caiman_obj.estimates.f

        # TODO: get slice number to know how many to pass to peak caller
        # filename = join(self.work_dir, '')
        # demo: a single image
        filename = Path(self.work_dir).joinpath("out") if not self.do_s2 else self.imm2_list[0]
        pc_obj1 = PeakCaller(data, filename)
        pc_obj1.Detrender_2()
        pc_obj1.Find_Peak()
        new_series = pc_obj1.Filter_Series('src/finalized_model.sav')
        pc_obj2 = PeakCaller(new_series, filename)
        pc_obj2.Detrender_2()
        pc_obj2.Find_Peak()
        pc_obj2.Print_ALL_Peaks()
        pc_obj2.Raster_Plot()
        pc_obj2.Histogram_Height()
        pc_obj2.Histogram_Time()
        pc_obj2.Correlation()
        # To save results, do something like this:
        pc_obj2.Synchronization()
        pc_obj2.Save_Result()
        self.pc_obj.append(pc_obj2)

    def run(self):
        msg = {
            'idx': self.queue_id, 'is_running': False, 'is_finished': False,
            'cm': True if self.do_s2 else False
        }
        # TODO: need to adjust imm1_list, imm2_list, according to which section is the first section
        if not self.do_s0:
            self.s1_root = self.input_root
            self.imm1_list = self.input_list
        if not self.do_s1:
            self.imm2_list = self.input_list
        if not self.do_s2 and self.do_s3:
            assert 'cmn_obj' in self.input_list

        start_time = perf_counter()
        if self.queue:
            msg['is_running'] = True
            self.queue.put(msg)
        # First, decide which section to start execute
        if self.do_s0:
            # Do cropping
            self.s0()
            self.done_s0 = True
        if self.do_s1:
            # Do stabilizer
            self.s1()
            self.done_s1 = True
        if self.do_s2:
            # CaImAn part
            start_time_caiman = time()
            self.s2()
            end_time_caiman = time()
            exec_t = end_time_caiman - start_time_caiman
            self.log_print(f"caiman part took {exec_t // 60}m {int(exec_t % 60)}s.")
            self.done_s2 = True

        # Peak_caller part
        if self.do_s3:
            self.s3()

        end_time = perf_counter()
        exec_time = end_time - start_time
        self.log_print(f"[INFO] pipeline.run() takes {exec_time // 60}m {int(exec_time % 60)}s.")
        if self.log is not None and not self.log.closed:
            self.log.close()
        if self.queue:
            msg['is_running'] = False
            msg['is_finished'] = True
            self.queue.put(msg)

    def load_setting(self, settings):
        self.QCimage_s2 = self.cache.joinpath(settings['QC']['s2_image'])

    def qc_s2(self, frame_idx, ROI_idx):
        from cv2 import resize
        frame_per_file = len(self.caiman_obj.input_files[0])
        image_idx = frame_idx // frame_per_file
        frame_idx %= frame_per_file
        ROI = np.reshape(self.caiman_obj.estimates.A[:, ROI_idx].toarray(), self.caiman_obj.dims, order='F')
        # ROI = np.array(ROI, dtype=np.uint8)
        w, h = ROI.shape
        new_w = int(200 * h / w)
        ROI = resize(ROI, (new_w, 200))
        y, x = np.unravel_index(ROI.argmax(), ROI.shape)
        image_raw = self.caiman_obj.input_files[image_idx][frame_idx]
        w_r, h_r = ROI.shape
        new_w_r = int(200 * h_r / w_r)
        image_raw = resize(image_raw, (new_w_r, 200))
        ROI_temp = ROI * 255 + image_raw
        ROI_temp = cv2.rectangle(ROI_temp, (x, y), (x + 15, y + 15), (255, 0, 0), 2)
        return ROI_temp

    # def qc_caiman_movie(self):
    #     if self.ij is None:
    #         try:
    #             self.ij = imagej.init(self.ijp, mode='interactive')
    #         except Exception as e:
    #             raise RuntimeError("ImageJ not initialized.")
    #     if self.QCimage_s2 is None:
    #         raise RuntimeError("QC image not found.")
    #     if self.caiman_obj.outpath_s2 == '':
    #         raise RuntimeError("No caiman output path.")
    #     elif not Path(self.outpath_s2).exists():
    #         raise RuntimeError(f"Caiman output path {self.outpath_s2} not found.")
    #
    #     # Start playing output movie
    #     dataset = self.ij.IJ.run("Bio-Formats Importer",
    #                              f"open={self.caiman_obj.outpath_s2} autoscale color_mode=Grayscale rois_import=[ROI manager] "
    #                              f"view=Hyperstack stack_order=XYCZT")
    #     # dataset = self.ij.io().open(self.outpath_s2)
    #     # self.ij.ui().show(dataset)


class QC:
    __slots__ = [
        'cmnobj_path',
        'cmn_obj',
        'movie',
        'current_frame',
        'qc_tab',
    ]

    def __init__(self, cmnobj_path=None, debug=False):
        if debug:
            self.cmnobj_path = None
            self.cmn_obj = None
            self.movie = None
            self.current_frame = 0
            self.qc_tab = None
        else:
            from pickle import load
            self.cmnobj_path = cmnobj_path
            with open(cmnobj_path, 'rb') as f:
                self.cmn_obj = load(f)
            self.movie = self.cmn_obj.input_files
            self.current_frame = 0

    def show_frame(self, frame_idx, image_idx=0) -> np.ndarray:
        frame = self.movie[image_idx][frame_idx]
        return frame


class CalciumZero:
    __slots__ = [
        'run_instance',
        'qc_instance',
    ]

    def __init__(self):
        self.run_instance = None
        self.qc_instance = None
