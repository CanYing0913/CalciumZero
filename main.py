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

    # print("[INFO] Starting section 0 - dense segmentation:")
    # image_crop, fname_crop = s0(work_dir, fname_in=file, margin=200, fname_out=None, save=True, debug=False)

    fname_crop = r"E:/work_dir/case1 Movie_57_crop.tif"  # for unit testing purpose
    print("[INFO] Starting section 1 - ImageJ Stabilizer:")
    image_sb = s1(work_dir, app_path, fname_crop)  # , *args)

    # print("[INFO] Starting section 2 - CaImAn:")
    # pass


if __name__ == "__main__":
    try:
        main(r"E:/work_dir", r"D:\CanYing\Fiji.app", r"E:/case1 Movie_57.tif")
    except:
        print("error")
