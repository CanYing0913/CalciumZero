from src_detection import s0
from src_stabilizer import s1
# from src_caiman import s2

import os


def main(work_dir: str, file: str):
    """

    Parameters:
        work_dir: Directory where all temporal data are stored.
        file: Selected raw input file path.
    """
    if not os.path.exists(work_dir):
        raise FileNotFoundError("Project Working directory not found.")
    print("[INFO] Starting section 0:")
    image = s0(work_dir, fname_in: str, margin: int, fname_out = None, save = True, debug = False)
    print("[INFO] Starting section 1:")
    image = s1(work_dir, app_path, fpath_in, fpath_out, *args)


if __name__ == "__main__":
    try:
        main("","")
    except:
        print()
