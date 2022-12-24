import sys
import time

from src import src_detection as s0
from src import src_stabilizer as s1
from src import src_caiman as s2
import argparse
import os
from pathlib import Path


def parse():
    desp = "Automated pipeline for CaImAn processing."
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('-ijp', '--imagej-path', type=str, metavar='ImageJ-Path', required=True,
                        help='Path to local Fiji ImageJ fiji folder.')
    parser.add_argument('-wd', '--work-dir', type=str, metavar='Work-Dir', required=True,
                        help='Path to a working folder where all intermediate and overall results are stored. If not '
                             'exist then will create one automatically.')
    parser.add_argument('-input', type=str, metavar='INPUT', required=True,
                        help='Path to input/inputs folder. If you have multiple inputs file, please place them inside a single folder.')
    parser.add_argument('-in1', type=str, metavar='INPUT1', required=False,
                        help='Path to raw input files, supplied to auto-cropping. If provided without absolute path, '
                             'it will assume the file exists inside working folder.')
    parser.add_argument('-margin', default=200, type=int, metavar='Margin', required=False,
                        help='Margin in terms of pixels for auto-cropping. Default to be 200.')
    parser.add_argument('-intermediate1', default=False, action="store_true", required=False,
                        help='Specified if want to begin execution starting with ImageJ stabilizer (skipping '
                             'auto-cropping).')
    parser.add_argument('-in2', type=str, metavar='', required=False,
                        help='Filename to save the intermediate result 1 within working folder. If specified, '
                             'auto-cropping will save cropped result to the filename specified, and stabilizer will '
                             'read from it. Please only provide with relative path, it assumes to save inside working '
                             'folder. If -intermediate1 is specified, the program will skip auto-cropping, and read '
                             'from filename specified to start stabilizer.')
    parser.add_argument('-intermediate2', default=False, action="store_true", required=False,
                        help='Specified if want to begin execution starting with caiman (skipping auto-cropping and '
                             'stabilizer).')
    parser.add_argument('-in3', type=str, metavar='', required=False,
                        help='Filename to save the intermediate result 2 within working folder. If specified, '
                             'stabilizer will save stabilized result to the filename specified, and caiman will read '
                             'from it. Please only provide with relative path, it assumes to save inside working '
                             'folder. If -intermediate2 is specified, the program will skip auto-cropping and '
                             'stabilizer, and read from filename specified to start caiman.')
    parser.add_argument('-out', type=str, metavar='', required=False,
                        help='Output filename (hdf5 file) for caiman output. If not specified, it will modify caiman '
                             'input filename to use as output filename. Please only provide with relative path, '
                             'it assumes to save inside working folder. It will also save numpy arrays as well as '
                             'other related debug files.')
    # ImageJ Stabilizer Parameters
    parser.add_argument('-ij_param', default=False, action="store_true", required=False,
                        help='Specified if want to customize ImageJ stabilizer parameters.')
    parser.add_argument('-ij_trans', default=0, type=int, required=False,
                        help='Transformation of ImageJ stabilizer parameter. You have to specify -ij_param to use it. '
                             'Default to translation, set it to 1 if want to set it to affine.')
    parser.add_argument('-ij_maxpl', default=1, type=float, required=False,
                        help='MAX_Pyramid_level of ImageJ stabilizer parameter. You have to specify -ij_param to use '
                             'it. Default to be 1.0.')
    parser.add_argument('-ij_upco', default=0.90, type=float, required=False,
                        help='update_coefficient of ImageJ stabilizer parameter. You have to specify -ij_param to use '
                             'it. Default to 0.90.')
    parser.add_argument('-ij_maxiter', default=200, type=int, required=False,
                        help='MAX_iteration of ImageJ stabilizer parameter. You have to specify -ij_param to use it. '
                             'Default to 200.')
    parser.add_argument('-ij_errtol', default=1E-7, type=float, required=False,
                        help='error_rolerance of ImageJ stabilizer parameter. You have to specify -ij_param to use '
                             'it. Default to 1E-7.')
    parser.add_argument('-log', type=bool, default=False, required=False,
                        help='True if enable logging for caiman part. Default to be false.')

    arguments = parser.parse_args()

    # Post-process arguments
    arguments.imagej_path = rf"{arguments.imagej_path}"
    arguments.work_dir = rf"{arguments.work_dir}"
    arguments.input = rf"{arguments.input}"
    if os.path.exists(arguments.input):
        if os.path.isdir(arguments.input):
            # A folder of multiple inputs
            arguments.input = [f for f in os.listdir(arguments.input) if '.tif' in f]
        else:
            # A folder of multiple inputs
            pass
    else:
        raise FileNotFoundError(f"file path {arguments.input} does not exist.")
    if hasattr(arguments, 'in1'):
        arguments.in1 = rf"{arguments.in1}"
    if hasattr(arguments, 'in2'):
        arguments.in2 = rf"{arguments.in2}" if arguments.in2 is not None else None
    if hasattr(arguments, 'in3'):
        arguments.in3 = rf"{arguments.in3}" if arguments.in3 is not None else None

    return arguments


def main(work_path: str, app_path: str, file: str, argv):
    """
    Main pipeline function.

    Parameters:
        work_path: Directory where all temporal data are stored.
        app_path: Path pointing to local Fiji ImageJ fiji folder.
        file: Selected raw input file path.
        argv: arguments parsed by argparse.
    """
    if not os.path.exists(work_path):
        print("[NOTE INFO]: Project Working directory does not exist. Attempting to create one...")
        try:
            Path(work_path).mkdir(parents=True, exist_ok=False)
        except OSError:
            print(f"[ERROR]: OSError detected. Please check if disk exists or privilege level satisfies.")
            exit(1)

    fname_crop = argv.in2
    if not argv.intermediate1 and not argv.intermediate2:
        print("[INFO] Starting section 0 - dense segmentation/auto-cropping:")
        image_crop, fname_crop = s0.s0(work_dir=work_path, fname_in=file, margin=200, fname_out=argv.in2, save=True, debug=False)
    else:
        print("[INFO] Skipping section 0 - dense segmentation/auto-cropping.")

    # fname_crop = r"E:/work_dir/case1 Movie_57_crop.tif"  # for unit testing purpose
    fpath_sb = argv.in3
    if not argv.intermediate2:
        print("[INFO] Starting section 1 - ImageJ Stabilizer:")
        # currently, ImageJ asks for output directory even with headless plugin (need to be verified from image.sc)
        # just prompt "click cancel" if problem persists.
        image_sb, fpath_sb = s1.s1(work_dir=work_path, app_path=app_path, fpath_in=fname_crop, fpath_out=argv.in3, argv=argv)
    else:
        print("[INFO] Skipping section 1 - ImageJ Stabilizer.")

    print("[INFO] Starting section 2 - CaImAn:")
    st = time.time()
    s2.s2(work_dir=work_path, fpath_in=fpath_sb, fpath_out=None, save=True, log=argv.log)
    print(f"[INFO] Section 2: caiman finished. Total execution time: {int(time.time() - st) / 60} m {(time.time() - st) % 60} s")


if __name__ == "__main__":
    st = time.time()
    # Assume parse() already takes care of handling arguments
    args = parse()
    ImageJ_path = args.imagej_path
    work_dir = args.work_dir
    margin = args.margin
    fpath_in1, fpath_in2, fpath_in3 = args.input, args.in2, args.in3
    fapth_out = args.out
    imm1, imm2 = args.intermediate1, args.intermediate2

    main(work_path=work_dir, app_path=ImageJ_path, file=fpath_in1, argv=args)
    print(f"total taken {time.time() - st} seconds")
    sys.exit(0)
