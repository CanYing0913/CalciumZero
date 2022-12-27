import os
from os.path import join, exists, isfile, isdir, basename, dirname
import sys
import argparse
from pathlib import Path
from time import sleep
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
    parser.add_argument('-clog', default=False, required=False,
                        help='True if enable logging for caiman part. Default to be false.')
    parser.add_argument('-csave', default=False, action='store_true',
                        help='True if want to save denoised movie. Default to be false.')

    return parser.parse_args()


def main():
    print("[INFO]: Collecting tasks before pipeline. Read the following info carefully.")
    print("[INFO]: Do NOT place your input file(s) under a disk on your computer otherwise mounting will fail.")
    arguments = parse()
    sleep(2)
    if not exists(arguments.input):
        raise FileNotFoundError(f"input path {arguments.input} does not exist.")
    mnt_path = input_path = arguments.input

    # detection if no need to change hierarchy (say already run once)
    pass_change = False
    if isfile(input_path):
        # single input path
        print("[NOTE]: You are running the pipeline with single input.")
        # 1 case: path to .../_.tif
        ppath_i = dirname(input_path)
        if basename(ppath_i) == "in":
            ppath_o = join(dirname(ppath_i), "out")
            if exists(ppath_o) and isdir(ppath_o):
                pass_change = True
                mnt_path = dirname(ppath_i)
    elif isdir(input_path):
        # input directory path
        print("[NOTE]: You are running the pipeline with a input folder.")
        # 2 cases: path to mnt/in/, or path to mnt/
        ppath_o = join(dirname(input_path), "out")
        if basename(input_path) == "in" and exists(ppath_o) and isdir(ppath_o):
            mnt_path = dirname(input_path)
            pass_change = True
            input_list = [f for f in os.listdir(input_path) if f[-4:] == ".tif"]
        else:
            # input path to mnt/
            if "in" in os.listdir(input_path) and "out" in os.listdir(input_path):
                ppath_i = join(input_path, "in")
                input_list = [f for f in os.listdir(ppath_i) if f[-4:] == ".tif"]
                ppath_o = join(input_path, "out")
                if isdir(ppath_i) and isdir(ppath_o):
                    # mnt_path already set
                    pass_change = True
    else:
        raise FileNotFoundError(f"Invalid input path {input_path}")

    # Get file hierarchy report
    if not pass_change:
        if isfile(input_path):
            print("Here is the hierarchy we will make:")
            sleep(1)
            print("\t" + mnt_path.removesuffix(basename(input_path)))
            print("\t|--->" + "mnt/")
            print("\t\t|--->in/")
            print(f"\t\t\t|--->{basename(input_path)}")
            print("\t\t|--->out/")
        elif isdir(input_path):
            input_list = [f for f in os.listdir(mnt_path) if f[-4:] == '.tif']
            print("Here is the hierarchy we will make:")
            sleep(1)
            print("\t" + mnt_path)
            print("\t\t|--->in/")
            for i in range(min(len(input_list), 2)):
                print(f"\t\t\t|--->{input_list[i]}")
            if len(input_list) > 2:
                print(f"\t\t\t|---> ...")
            print("\t\t|--->out/")

        # Create mounting hierarchy
        sleep(2)
        ans = ''
        while ans.casefold() != 'y'.casefold() and ans.casefold() != 'n'.casefold():
            ans = input('Are you sure you want to make these changes? [Y/N]:')
        if ans.casefold() == 'n'.casefold():
            sys.exit(1)
        print(f"[INFO]: Creating mounting hierarchy...", end='')
        try:
            if isfile(input_path):
                mnt_path = join(mnt_path.removesuffix(basename(input_path)), 'mnt')
                Path(mnt_path).mkdir(parents=False, exist_ok=False)
                Path(join(mnt_path, "in")).mkdir(parents=False, exist_ok=True)
                shutil.move(input_path, join(mnt_path, "in", basename(input_path)))
            elif isdir(input_path):
                # mnt_path already set
                input_list = [f for f in os.listdir(mnt_path) if f[-4:] == '.tif']
                Path(join(mnt_path, "in")).mkdir(parents=False, exist_ok=True)
                for f in input_list:
                    shutil.move(join(mnt_path, f), join(mnt_path, "in", f))
            Path(join(mnt_path, "out")).mkdir(parents=False, exist_ok=True)
            print('Done')
        except:
            print(f"{sys.exc_info()[0]} occurred during changing hierarchy. Program exiting.")
            sys.exit(1)
    else:
        print(f"[INFO]: no need to change hierarchy to mount.")
        sleep(1)

    # Handle other parameters to pass to docker run command
    args_dict = vars(arguments)
    arg = ''
    for key, value in args_dict.items():
        if key == "input":
            arg += "-input "
            for f in input_list:
                arg += f"{f} "
        else:
            arg += f"-{key} {value} "

    # TODO: get container name
    container_name = 'test'
    # TODO: get image name
    pass
    cmd = f'docker run --name {container_name} -v "{mnt_path}":/tmp/mnt -i -t pipeline {arg}'
    print(cmd)
    # input("press enter to start docker run")
    # os.system(cmd)


if __name__ == "__main__":
    main()
