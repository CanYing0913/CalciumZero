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

# Retrieve source
from src_detection import *


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
    parser.add_argument('-ijp', '--imagej-path', type=str, metavar='ImageJ-Path', required=True,
                        help='Path to local Fiji ImageJ fiji folder.')
    parser.add_argument('-wd', '--work-dir', type=str, metavar='Work-Dir', required=True,
                        help='Path to a working folder where all intermediate and overall results are stored. If not '
                             'exist then will create one automatically.')
    parser.add_argument('-input', type=str, metavar='INPUT', required=True,
                        help='Path to input/inputs folder. If you have multiple inputs file, please place them inside '
                             'a single folder. If you only have one input, either provide direct path or the path to'
                             ' the folder containing the input(without any other files!)')
    parser.add_argument('--skip-0', default=False, action="store_true", required=False, help='Skip segmentation and '
                                                                                             'cropping if specified.')
    parser.add_argument('--skip-1', default=False, action="store_true", required=False, help='Skip stabilizer if '
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
    parser.add_argument('-log', type=bool, default=False, required=False,
                        help='True if enable logging for caiman part. Default to be false.')
    # Parse the arguments
    arguments = parser.parse_args()
    # Post-process arguments
    # ImageJ path and param
    arguments.imagej_path = Path(arguments.imagej_path)
    if not Path.exists(arguments.imagej_path):
        if not arguments.skip_1:
            raise OSError(f"[ERROR]: ImageJ path does not exist: {arguments.imagej_path}")
    arguments.ij_params = {
        'Transformation': arguments.ij_trans,
        'MAX_Pyramid_level': arguments.ij_maxpl,
        'update_coefficient': arguments.ij_upco,
        'MAX_iteration': arguments.ij_maxiter,
        'error_tolerance': arguments.ij_errtol
    }
    # work_dir path
    arguments.work_dir = Path(arguments.work_dir)
    if not Path.exists(arguments.work_dir):
        print(f"Working directory {arguments.work_dir} does not exist. Attempting to create one.")
        try:
            Path(arguments.work_dir).mkdir(parents=True, exist_ok=False)
        except OSError:
            print(f"[ERROR]: OSError detected. Please check if disk exists or privilege level satisfies.")
            exit(1)
    # input folder path
    arguments.input = Path(arguments.input)
    if Path.exists(arguments.input):
        if os.path.isdir(arguments.input):
            # Path to a folder of multiple inputs.
            arguments.input_root = arguments.input
            arguments.input = [f for f in os.listdir(str(arguments.input)) if '.tif' in f]
        else:
            # Path to a single input file.
            arguments.input_root = os.path.basename(arguments.input)
            if '.tif' in arguments.input:
                arguments.input = [arguments.input]
    else:
        raise FileNotFoundError(f"[ERROR]: Input file path {arguments.input} does not exist.")
    return arguments


class pipeline(object):
    def __init__(self):
        # Control sequence
        self.skip_0 = self.skip_1 = False
        self.work_dir = ''
        # Segmentation and cropping related variables
        self.input_root = ''
        self.input_list = []
        self.margin = 0
        # ImageJ stabilizer related variables
        self.ij = None
        self.s1_params = {}
        # CaImAn related variables
        self.caiman_obj = None

    def parse(self):
        # Retrieve calling parameters
        arguments = parse()
        # Use parameters to set up pipeline global info
        # Control releated
        self.work_dir = arguments.work_dir
        self.skip_0 = arguments.skip_0
        self.skip_1 = arguments.skip_1
        # Segmentation and cropping related variables
        self.input_root = arguments.input_root
        self.input_list = arguments.input
        self.margin = arguments.margin
        # ImageJ stabilizer related variables
        self.ij = imagej.init(arguments.imagej_path, mode='headless')
        print(f"ImageJ initialized with version {self.ij.getVersion()}.")
        self.s1_params = arguments.ij_params
        # CaImAn related variables
        # TODO: add caiman parameters
        pass
        # TODO: collect task report
        print(rf"******Tasks TODO list******")
        print(rf"")

    def s0(self, debug=False):
        def ps0(text: str):
            print(f"***[S0 - Detection]: {text}")
        # TODO: segmentation and cropping
        generator = get_image(self.input_list, self.input_root)
        fname_i, image_i = generator.__next__()
        ps0(f"Reading input {fname_i} with shape {image_i.shape}.")
        # Process input
        image_seg_o, th_l = denseSegmentation(image_i, debug)
        x1, y1, x2, y2 = find_bb_3D_dense(image_seg_o, debug)
        image_crop_o = apply_bb_3D(image_i, (x1, y1, x2, y2), self.margin)
        # Handle output path
        fname_seg_o = fname_i[fname_i.rfind("/") + 1: fname_i.rfind(".tif")] + '_seg.tif'
        fname_crop_o = fname_i[fname_i.rfind("/") + 1: fname_i.rfind(".tif")] + '_crop.tif'
        fname_seg_o = os.path.join(self.work_dir, fname_seg_o)
        fname_crop_o = os.path.join(self.work_dir, fname_crop_o)
        ps0(f"Using paths: {fname_seg_o} and {fname_crop_o} to save intermediate results.")
        tifffile.imwrite(fname_seg_o, image_seg_o)
        tifffile.imwrite(fname_crop_o, image_crop_o)
        return image_crop_o, fname_crop_o

    def s1(self):
        # TODO: stabilizer
        pass

    def s2(self):
        # TODO: caiman
        pass

    def s3(self):
        # TODO: peak_caller
        pass


def main():
    testobj = pipeline()
    testobj.parse()
    
    # testobj.s0()
    # testobj.s0()
    # testobj.s0()


if __name__ == '__main__':
    main()
