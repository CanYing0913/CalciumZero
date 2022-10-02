from src_detection import s0
from src_stabilizer import s1
# from src_caiman import s2

import os


def main(work_dir: str, app_path: str, file: str):
    """

    Parameters:
        work_dir: Directory where all temporal data are stored.
        file: Selected raw input file path.
    """
    if not os.path.exists(work_dir):
        raise FileNotFoundError("Project Working directory not found.")
    print("[INFO] Starting section 0 - dense segmentation:")
    image_crop, fname_out = s0(work_dir, fname_in=file, margin=200, fname_out=None, save=True, debug=False)
    print("[INFO] Starting section 1 - ImageJ Stabilizer:")
    image_sb = s1(work_dir, app_path, fname_out, fpath_out=None)  # , *args)
    print("[INFO] Starting section 2 - CaImAn:")



if __name__ == "__main__":
    try:
        main("", "")
    except:
        print()
