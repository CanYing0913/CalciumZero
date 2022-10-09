from src_detection import s0
from src_stabilizer import s1
from src_caiman import s2
import argparse
import os


def parse():
    desp = "Automated pipeline for caiman processing."
    parser = argparse.ArgumentParser(description=desp)
    parser.add_argument('-ijp', '--imagej-path', type=str, metavar='', required=True,
                        help='Path to local Fiji ImageJ fiji folder.')
    parser.add_argument('-wd', '--work-dir', type=str, metavar='', required=True,
                        help='Path to a working folder where all intermediate and overall results are stored. If not '
                             'exist then will create one automatically.')
    parser.add_argument('-in1', type=str, metavar='', required=True,
                        help='Path to raw input files, supplied to auto-cropping. If provided without absolute path, '
                             'it will assume the file exists inside working folder.')
    parser.add_argument('-intermediate1', type=bool, default=False, required=False,
                        help='Specified if want to begin execution starting with ImageJ stabilizer (skipping '
                             'auto-cropping).')
    parser.add_argument('-in2', type=str, metavar='', required=False,
                        help='Filename to save the intermediate result 1 within working folder. If specified, '
                             'auto-cropping will save cropped result to the filename specified, and stabilizer will '
                             'read from it. Please only provide with relative path, it assumes to save inside working '
                             'folder. If -intermediate1 is specified, the program will skip auto-cropping, and read '
                             'from filename specified to start stabilizer.')
    parser.add_argument('-intermediate2', type=bool, default=False, required=False,
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

    arguments = parser.parse_args()
    # TODO: post-process arguments

    return arguments


def main(work_dir: str, app_path: str, file: str):
    """
    Main pipeline function.

    Parameters:
        work_dir: Directory where all temporal data are stored.
        app_path: Path pointing to local Fiji ImageJ fiji folder.
        file: Selected raw input file path.
    """
    if not os.path.exists(work_dir):
        raise FileNotFoundError("Project Working directory not found.")

    # print("[INFO] Starting section 0 - dense segmentation:")
    # image_crop, fname_crop = s0(work_dir, fname_in=file, margin=200, fname_out=None, save=True, debug=False)

    fname_crop = r"E:/work_dir/case1 Movie_57_crop.tif"  # for unit testing purpose
    print("[INFO] Starting section 1 - ImageJ Stabilizer:")
    # currently, ImageJ asks for output directory even with headless plugin (need to be verified from image.sc)
    # just prompt "click cancel" if problem persists.
    image_sb, fpath_sb = s1(work_dir, app_path, fname_crop)  # , *args)

    # print("[INFO] Starting section 2 - CaImAn:")
    s2(app_path, fpath_in=fpath_sb, fpath_out=None)
    # pass


if __name__ == "__main__":
    # Assume parse() already takes care of handling arugments
    args = parse()
    ImageJ_path = args.ijp
    work_dir = args.wd
    fpath_in1, fpath_in2, fpath_in3 = args.in1, args.in2, args.in3
    fapth_out = args.out
    imm1, imm2 = args.intermediate1, args.intermediate2

    try:
        main(r"E:/work_dir", r"D:\CanYing\Fiji.app", r"E:/case1 Movie_57.tif")
    except:
        print("error")
