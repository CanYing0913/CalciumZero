"""
Source file for pipeline in general, OOP workflow
Last edited on July 17, 2024
Copyright @ Yian Wang (canying0913@gmail.com) - 2024
"""
from multiprocessing import Pool, Queue
from time import perf_counter
from typing import Optional, Tuple, List

from src.src_caiman import *
from src.src_detection import *
# from src.src_peak_caller import PC
from src.src_stabilizer import run_plugin
from src.utils import iprint


class Pipeline(object):
    def __init__(
            self,
            queue: Optional[Queue] = None,
            queue_id: int = 0,
            log_queue: Optional[Queue] = None,
    ):
        # GUI-related
        self.msg_queue = queue
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
        self.work_dir = ''
        self.process = 2
        # Segmentation and cropping related variables
        self.do_s0 = False
        self.input_root = ''
        self.input_list = []
        self.done_s0 = False
        # ImageJ stabilizer related variables
        self.do_s1 = False
        self.ijp = Path(__file__).parent.parent.joinpath('Fiji.app')
        self.done_s1 = False
        # CaImAn related variables
        self.do_s2 = False
        self.caiman_obj = None
        self.clog = False  # True
        self.csave = False
        self.done_s2 = False
        self.outpath_s2 = ''
        self.cmobj_path = ''
        # Peak Caller related
        self.do_s3 = False
        self.pc_obj = []

    def log(self, txt: str):
        if self.log_queue:
            iprint(txt, log_queue=self.log_queue)
        elif self.logger:
            iprint(txt, logger=self.logger)
        else:
            iprint(txt)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.log(f'the requested key {key} does not exist.')
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

    def s0(self, file_list) -> Tuple[List[Path], List[Path]]:
        start_time = perf_counter()
        # Segmentation and cropping
        # Scanning for bounding box for multiple input
        filenames_seg = []
        filenames_crop = []
        with Pool(processes=self.process) as pool:
            # filenames = [str(Path(self.input_root).joinpath(fname)) for fname in self.input_list]
            filenames = file_list
            results = pool.map(scan, filenames)
        x1, y1, x2, y2 = reduce_bbs(results)
        for idx, file in enumerate(filenames):
            filename_seg = Path(self.work_dir).joinpath(Path(file).stem + "_seg.tif")
            tifffile.imwrite(str(filename_seg), results[idx][-1])
            filenames_seg.append(filename_seg)
        finalxcntss = [result[4] for result in results]

        # Apply the uniform bb one-by-one to each input image
        for idx, filename_in in enumerate(filenames):
            image_i = tifffile.imread(filename_in)
            finalxcnts = finalxcntss[idx]
            image_crop_o = apply_bb_3d(image_i, (x1, y1, x2, y2), self.params_dict['crop']['threshold'], finalxcnts)
            # Handle output path
            filename_out = Path(self.work_dir).joinpath(Path(filename_in).stem + "_crop.tif")
            self.log(f"*[Detection]: Using paths: {filename_out} to save cropped result.")
            # Save imm1 data to files
            tifffile.imwrite(str(filename_out), image_crop_o)
            filenames_crop.append(filename_out)
        self.done_s0 = True
        end_time = perf_counter()
        self.log(f"*[Detection]: Detection finished in {int(end_time - start_time)} s.")
        return filenames_crop, filenames_seg

    def s1(self, file_list: List[Path]) -> List[Path]:
        # ImageJ Stabilizer
        self.log(f"*[Stabilizer]: Stabilizer Starting.")
        results = []
        start_time = perf_counter()
        idx = 0
        self.log(f"*[Stabilizer]: Using files {file_list} for stabilizer.")
        while idx < len(file_list):
            imm1_list = [file_list[idx + i] for i in range(self.process) if idx + i < len(file_list)]
            idx += self.process
            with Pool(processes=len(file_list)) as pool:
                results = pool.starmap(run_plugin,
                                       [(
                                           self.ijp, str(file), self.work_dir,
                                           self.params_dict['stabilizer'])
                                           for file in imm1_list])
        end_time = perf_counter()
        self.log(f"*[Stabilizer]: Stabilizer finished in {int(end_time - start_time)} s.")
        self.log(f'*[Stabilizer]: Stabilizer result are placed in {results}.')
        return results

    def s2(self, file_list) -> Tuple[Path, Optional[List[Path]]]:
        self.cmobj_path = Path(self.work_dir).joinpath("cmn_obj.cmobj")

        start_time = perf_counter()
        if self.clog:
            self.log(f"*[CaImAn]: caiman logging enabled.")
            logging.basicConfig(
                format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                level=logging.DEBUG)
        filenames_in = file_list
        self.outpath_s2 = [Path(self.work_dir).joinpath(Path(f).stem + '_caiman.tif') for f in filenames_in]
        self.log(f"*[CaImAn]: caiman sets input: {filenames_in}, output path: {self.outpath_s2}")
        self.params_dict['caiman']['mc_dict']['fnames'] = filenames_in
        opts = params.CNMFParams(params_dict=self.params_dict['caiman']['mc_dict'])
        # Motion Correction
        if self.params_dict['caiman']['mc_dict']['motion_correct']:
            self.log(f"*[CaImAn]: Running motion correction...")
            # do motion correction rigid
            mc = MotionCorrect(filenames_in, dview=None, **opts.get_group('motion'))
            mc.motion_correct(save_movie=True)
            fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
            if pw_rigid:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                             np.max(np.abs(mc.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(mc.total_template_rig)  # % plot template
                # plt.subplot(1, 2, 2)
                # plt.plot(mc.shifts_rig)  # % plot rigid shifts
                # plt.legend(['x shifts', 'y shifts'])
                # plt.xlabel('frames')
                # plt.ylabel('pixels')
                # plt.show()

            bord_px = 0 if border_nan == 'copy' else bord_px
            fname_mmap = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            self.log(f"*[CaImAn]: Motion correction skipped.")
            bord_px = 0
            fname_mmap = cm.save_memmap([str(filename) for filename in filenames_in],
                                        base_name='memmap_', order='C', border_to_0=0, dview=None)
        self.log(f"*[CaImAn]: mmap file saved to {fname_mmap}")

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
        self.log(f"*[CaImAn]: Running CNMF-E...")
        start_time_cnmf = perf_counter()
        cnm = cnmf.CNMF(n_processes=8, dview=None, Ain=Ain, params=opts)
        cnm.fit(images)
        setattr(cnm, 'images', images)
        end_time_cnmf = perf_counter()
        exec_time_cnmf = end_time_cnmf - start_time_cnmf
        self.log(f"*[CaImAn]: cnmf takes {exec_time_cnmf // 60}m, {int(exec_time_cnmf % 60)}s to complete.")
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

        self.log('*[CaImAn]:  ***** ')
        self.log(f'*[CaImAn]: Number of total components:  {len(cnm.estimates.C)}')
        self.log(f'*[CaImAn]: Number of accepted components: {len(cnm.estimates.idx_components)}')

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
            # generate boolean indexing
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
        self.log(f"*[CaImAn]: frames: {frames}")
        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=frames)
        setattr(cnm, 'dims', dims)
        # Save input files to *.cmobj file
        setattr(cnm, 'input_files', [tifffile.imread(fname) for fname in filenames_in])
        setattr(cnm, 'bord_px', bord_px)
        self.caiman_obj = cnm
        # reconstruct denoised movie
        if self.csave:
            denoised = cm.movie(cnm.estimates.A.dot(cnm.estimates.C)).reshape(dims + (-1,), order='F').transpose(
                [2, 0, 1])
            denoised.save(str(self.outpath_s2))
            self.log(f"*[CaImAn]: caiman denoised movie saved to {self.outpath_s2}")
        with open(self.cmobj_path, "wb") as f:
            pickle.dump(cnm, f)
            self.log(f"*[CaImAn]: object cnm dumped to {self.cmobj_path}.")
        end_time = perf_counter()
        exec_time = end_time - start_time
        self.log(f"*[CaImAn]: caiman finished in {exec_time // 60}m, {int(exec_time % 60)} s.")
        return self.cmobj_path, self.outpath_s2

    def s3(self, file_list):
        def ps3(txt: str):
            self.log(f"*[PeakCaller]: {txt}")
        # peak_caller
        # slice_num = _
        if not self.do_s2:
            pass
        data = self.caiman_obj.estimates.f

        pass

    def run(self):
        msg = {
            'idx': self.queue_id, 'is_running': False, 'is_finished': False,
            'cm': True if self.do_s2 else False
        }
        # TODO: need to adjust imm1_list, imm2_list, according to which section is the first section
        if not self.do_s2 and self.do_s3:
            assert 'cmn_obj' in self.input_list

        start_time = perf_counter()
        if self.msg_queue:
            msg['is_running'] = True
            self.msg_queue.put(msg)
        # First, decide which section to start execute
        file_list = self.input_list
        if self.do_s0:
            # Do cropping
            file_list, _ = self.s0(file_list)
            self.done_s0 = True
        if self.do_s1:
            # Do stabilizer
            file_list = self.s1(file_list)
            self.done_s1 = True
        if self.do_s2:
            # CaImAn part
            start_time_caiman = time()
            file_list = self.s2(file_list)
            end_time_caiman = time()
            exec_t = end_time_caiman - start_time_caiman
            self.log(f"caiman part took {exec_t // 60}m {int(exec_t % 60)}s.")
            self.done_s2 = True

        # Peak_caller part
        if self.do_s3:
            self.s3(file_list)

        end_time = perf_counter()
        exec_time = end_time - start_time
        self.log(f"[INFO] pipeline.run() takes {exec_time // 60}m {int(exec_time % 60)}s.")
        if self.msg_queue:
            msg['is_running'] = False
            msg['is_finished'] = True
            self.msg_queue.put(msg)


class QC:
    __slots__ = [
        'cmnobj_path',
        'cmn_obj',
        'data',
        'movies',
        'current_frame',
        'qc_tab',
    ]

    def __init__(self, cmnobj_path=None, debug=False):
        if debug:
            self.cmnobj_path = None
            self.cmn_obj = None
            self.data = None
            self.movies = None
            self.current_frame = 0
            self.qc_tab = None
        else:
            self.cmnobj_path = cmnobj_path
            self._load_data()

    def _load_data(self):
        from pickle import load
        with open(self.cmnobj_path, 'rb') as f:
            self.cmn_obj = load(f)
        self.data = self.cmn_obj
        self.movies = self.cmn_obj.input_files
        self.current_frame = 0

    @property
    def n_images(self):
        return len(self.movies)

    def image_shape(self, image_idx: int = 0):
        return self.movies[image_idx].shape

    @property
    def n_ROIs(self):
        return self.data.estimates.A.shape[1]

    def show_frame(self, image_idx: int = 0, frame_idx: int = 0, ROI_idx: Optional[int] = None):
        from cv2 import resize
        # 1. Check image index, and frame index.
        assert image_idx < len(self.movies), f"Image index {image_idx} out of range."
        frame_range = len(self.movies[image_idx])
        assert frame_idx < frame_range, \
            f"Frame index {frame_idx} out of range of image{image_idx}:{frame_range}."
        # 2. Scale image to 200 pixels height
        image_raw = self.movies[image_idx][frame_idx]
        w_r, h_r = image_raw.shape
        new_w_r = int(200 * h_r / w_r)
        image_raw = resize(image_raw, (new_w_r, 200))
        # 3. ROI handling
        if ROI_idx:
            # 3.1. Check ROI index
            ROI_range = self.data.estimates.A.shape[1]
            assert ROI_idx < ROI_range, f"ROI index {ROI_idx} out of range:{ROI_range}."
            # 3.2. Get ROI, scale it to 200 pixels height
            ROI = np.reshape(self.data.estimates.A[:, ROI_idx].toarray(), self.data.dims, order='F')
            # ROI = np.array(ROI, dtype=np.uint8)
            w_roi, h_roi = ROI.shape
            new_w_roi = int(200 * h_roi / w_roi)
            ROI = resize(ROI, (new_w_roi, 200))
            y, x = np.unravel_index(ROI.argmax(), ROI.shape)
            # 3.3. Overlay ROI on image and draw a box around the ROI
            ROI_temp = ROI * 255 + image_raw
            movie_with_ROIbox = cv2.rectangle(ROI_temp, (x, y), (x + 15, y + 15), (255, 0, 0), 2)
            return movie_with_ROIbox
        else:
            return image_raw


class CalciumZero:
    __slots__ = [
        'run_instance',
        'qc_instance',
    ]

    def __init__(self):
        self.run_instance = None
        self.qc_instance = None
