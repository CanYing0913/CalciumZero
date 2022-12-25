"""
Source file for pipeline in general, OOP workflow
Last edited on Dec.19 2022
Copyright Yian Wang (canying0913@gmail.com) - 2022
"""
import argparse
import os
from math import sqrt
from random import randint
from time import time
from pathlib import Path
import numpy as np
import tifffile
import seaborn
import imagej
from tqdm import tqdm
from src.src_peak_caller import PeakCaller

# Retrieve source
from src.src_detection import dense_segmentation, find_bb_3D_dense, apply_bb_3D
from src.src_stabilizer import print_param, run_stabilizer


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
    parser.add_argument('-no_log', default=True, action='store_false',
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
    parser.add_argument('-skip_0', default=False, action="store_true", required=False, help='Skip segmentation and '
                                                                                             'cropping if specified.')
    parser.add_argument('-skip_1', default=False, action="store_true", required=False, help='Skip stabilizer if '
                                                                                             'specified.')
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
    parser.add_argument('-c_log', type=bool, default=False, required=False,
                        help='True if enable logging for caiman part. Default to be false.')
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
            arguments.input_root = arguments.input.removesuffix(temp)
            arguments.input = [temp]
    else:
        raise FileNotFoundError(f"[ERROR]: Input file path {arguments.input} does not exist.")
    return arguments


class pipeline(object):
    def __init__(self):
        # Control sequence
        self.skip_0 = self.skip_1 = False
        self.work_dir = ''
        self.log = None
        # Segmentation and cropping related variables
        self.input_root = ''
        self.input_list = []
        self.margin = 0
        self.imm1_list = []  # Intermediate result list 1, relative path
        # ImageJ stabilizer related variables
        self.ij = None
        self.s1_params = []
        self.s1_root = ''
        self.imm2_list = []  # Intermediate result list 2, relative path
        # CaImAn related variables
        self.caiman_obj = None
        self.s2_root = ''
        # Peak Caller related
        self.pc_obj = None

    def pprint(self, txt: str):
        """
        Customized print function that both print to stdout and log file
        """
        print(txt)
        self.log.write(txt + '\n')

    def parse(self):
        # Retrieve calling parameters
        arguments = parse()
        # Use parameters to set up pipeline global info
        # Control related
        self.work_dir = arguments.work_dir
        self.skip_0 = arguments.skip_0
        self.skip_1 = arguments.skip_1
        if arguments.no_log:
            log_path = os.path.join(self.work_dir, 'log.txt')
            self.log = open(log_path, 'w')
            self.pprint(f"log file is stored @ {log_path}")
        # Segmentation and cropping related variables
        self.input_root = arguments.input_root
        self.input_list = arguments.input
        self.margin = arguments.margin
        # ImageJ stabilizer related variables
        self.ij = imagej.init(arguments.imagej_path, mode='headless')
        self.pprint(f"ImageJ initialized with version {self.ij.getVersion()}.")
        self.s1_params = arguments.ij_params
        # CaImAn related variables
        # TODO: add caiman parameters
        pass
        # TODO: add peak_caller parameters
        pass
        # Get control params to determine dest list
        # TODO: need extra care for caiman mmap generation
        # Must only specify one skip
        assert self.skip_0 is False or self.skip_1 is False, "Duplicate skip param specified."
        self.s1_root = self.s2_root = self.work_dir
        if self.skip_0:
            self.s1_root = self.input_root
            self.imm1_list = self.input_list
        elif self.skip_1:
            self.s2_root = self.input_root
            self.imm2_list = self.input_list
        return None

    def s0(self, debug=False):
        """
        Function to run segmentation, detection and cropping.

        Parameters:
            debug: Used for debugging purposes.
        """

        def ps0(text: str):
            self.pprint(f"***[S0 - Detection]: {text}")

        # TODO: segmentation and cropping
        # Scanning for bounding box for multiple input
        x1 = y1 = float('inf')
        x2 = y2 = -1
        for fname_i in self.input_list:
            image_i = tifffile.imread(os.path.join(self.input_root, fname_i))
            ps0(f"Reading input {fname_i} with shape {image_i.shape}.")
            # Process input
            image_seg_o, th_l = dense_segmentation(image_i, debug)
            x1_, y1_, x2_, y2_ = find_bb_3D_dense(image_seg_o, debug)
            if not debug:
                del th_l, image_seg_o
            x1 = min(x1, x1_)
            y1 = min(y1, y1_)
            x2 = max(x2, x2_)
            y2 = max(y2, y2_)
        del x1_, y1_, x2_, y2_
        if debug:
            ps0(f"Bounding box found with x1,y1,x2,y2: {x1, y1, x2, y2}")
        # Apply the uniform bb one-by-one to each input image
        for fname_i in self.input_list:
            image_i = tifffile.imread(os.path.join(self.input_root, fname_i))
            image_crop_o = apply_bb_3D(image_i, (x1, y1, x2, y2), self.margin)
            # Handle output path
            fname_crop_root = fname_i.removesuffix('.tif') + '_crop.tif'
            fname_crop_o = os.path.join(self.work_dir, fname_crop_root)
            ps0(f"Using paths: {fname_crop_o} to save cropped result.")
            # Save imm1 data to files
            tifffile.imwrite(fname_crop_o, image_crop_o)
            self.imm1_list.append(fname_crop_root)
        return

    def s1(self):
        def ps1(text: str):
            self.pprint(f"***[S1 - ImageJ stabilizer]: {text}")

        # TODO: ImageJ Stabilizer
        # TODO: select one file in self.imm_list1
        fname_i = self.imm1_list.pop()  # ''
        fname_i = os.path.join(self.s1_root, fname_i)
        ps1(f"Opening image at path {fname_i}...")
        imp = self.ij.IJ.openImage(fname_i)
        # Get ImageJ Stabilizer Parameters
        print_param(self.s1_params, ps1)
        # Start stabilizer
        ps1("Starting stabilizer in headless mode...")
        run_stabilizer(self.ij, imp, self.s1_params, ps1)
        # Set output path
        temp = os.path.basename(fname_i)
        fname_o = temp.removesuffix('.tif') + '_stabilized.tif'
        fname_o = os.path.join(self.work_dir, fname_o)
        ps1(f"Using output name: {fname_o}")
        # Save output and update imm_list2
        self.ij.IJ.saveAs(imp, "Tiff", fname_o)
        imp.close()
        self.imm2_list.append(fname_o)
        return

    def s2(self):
        # TODO: caiman
        pass

    def s3(self):
        # TODO: peak_caller
        data = self.caiman_obj.estimates.C[:92, :]
        filename = ''  # TODO
        self.pc_obj = PeakCaller(data, filename)
        self.pc_obj.Detrender_2()
        self.pc_obj.Find_Peak()
        # The above code generates a PeakCaller object with peaks detected
        self.pc_obj.Print_ALL_Peaks()
        self.pc_obj.Raster_Plot()
        self.pc_obj.Histogram_Height()
        self.pc_obj.Histogram_Time()
        self.pc_obj.Correlation()
        # To save results, do something like this:
        self.pc_obj.Synchronization()
        self.pc_obj.Save_Result()

    def run(self):
        # TODO: collect task report
        self.pprint(rf"******Tasks TODO list******")
        self.pprint(rf"")
        start_time = time()
        # First, decide which section to start execute
        skip0, skip1 = self.skip_0, self.skip_1
        if not skip0:
            # Do cropping
            self.s0(False)
        if not skip1:
            # Do stabilizer
            # TODO: pipeline stabilizer to make it working all the time.
            self.s1()
        # CaImAn part
        pass
        # Peak_caller part
        pass
        end_time = time()
        exec_t = end_time - start_time
        self.pprint(f"[INFO] pipeline.run() takes {exec_t // 60}m {int(exec_t % 60)}s to run in total.")
        self.log.close()


def main():
    testobj = pipeline()
    testobj.parse()

    # Note: current testing methodology is WRONG
    testobj.run()
    # testobj.s0()
    # testobj.s1()
    # testobj.s1()
    # testobj.s0()


if __name__ == '__main__':
    main()
    exit(0)
