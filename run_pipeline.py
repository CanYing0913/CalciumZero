import os
from os.path import join, exists, isfile, isdir, basename
import sys
import argparse
from pathlib import Path
import shutil


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, type=str, help='Path to your input (folder).')
    # pass param to main pipeline function
    parser.add_argument('-no_log', default=False, action='store_true',
                        help='Specified if do not want to have terminal printouts saved to a separate file.')
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

    return parser.parse_args()


def main():
    print("[INFO] Collecting tasks before pipeline. Read the following info carefully.")
    arguments = parse()
    if not exists(arguments.input):
        raise FileNotFoundError(f"input path {arguments.input} does not exist.")
    mnt_path = input_path = arguments.input

    # TODO: Get file hierarchy report
    # TODO: add detection if no need to change hierarchy (say already run once)
    if isfile(input_path):
        pass
    elif isdir(input_path):
        pass
    else:
        raise FileNotFoundError(f"Invalid input path {input_path}")
    ans = ''
    while ans.casefold() != 'y'.casefold() and ans.casefold() != 'n'.casefold():
        ans = input('Are you sure you want to make there changes? [Y/N]:')
    if ans.casefold() == 'n'.casefold():
        sys.exit(1)

    # Create mounting hierarchy
    if isfile(input_path):
        # single input path
        print("[NOTE]: You are running the pipeline with single input.")
        mnt_path = join(mnt_path.removesuffix(basename(input_path)), 'mnt')
        Path(mnt_path).mkdir(parents=False, exist_ok=False)
        Path(join(mnt_path, "in")).mkdir(parents=False, exist_ok=True)
        shutil.move(input_path, join(mnt_path, "in", basename(input_path)))
    elif isdir(input_path):
        # input directory path
        print("[NOTE]: You are running the pipeline with a input folder.")
        # mnt_path already set
        input_list = [f for f in os.listdir(mnt_path) if f[-4:] == '.tif']
        Path(join(mnt_path, "in")).mkdir(parents=False, exist_ok=True)
        for f in input_list:
            shutil.move(join(mnt_path, f), join(mnt_path, "in", f))
    Path(join(mnt_path, "out")).mkdir(parents=False, exist_ok=True)
    # Handle other parameters
    args_dict = vars(arguments)
    arg = ''
    for key, value in args_dict.items():
        arg += f"-{key} {value} "

    # TODO: get container name
    container_name = 'test'
    # TODO: get image name
    pass
    cmd = f'docker run --name {container_name} -v "{mnt_path}":/tmp/mnt -i -t pipeline {arg}'
    print(cmd)
    input("press enter to start docker run")
    # os.system(cmd)


if __name__ == "__main__":
    main()
