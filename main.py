import sys
from time import time

# from src import src_detection as s0
# from src import src_stabilizer as s1
# from src import src_caiman as s2
from src import src
import os
from pathlib import Path


def main():
    """
    Main pipeline function.
    """
    pipeline = src.pipeline()
    pipeline.parse()
    # if not os.path.exists(work_path):
    #     print("[NOTE INFO]: Project Working directory does not exist. Attempting to create one...")
    #     try:
    #         Path(work_path).mkdir(parents=True, exist_ok=False)
    #     except OSError:
    #         print(f"[ERROR]: OSError detected. Please check if disk exists or privilege level satisfies.")
    #         exit(1)
    #
    # fname_crop = argv.in2
    # if not argv.intermediate1 and not argv.intermediate2:
    #     print("[INFO] Starting section 0 - dense segmentation/auto-cropping:")
    #     image_crop, fname_crop = s0.s0(work_dir=work_path, fname_in=file_list, margin=200, save=True, debug=False)
    # else:
    #     print("[INFO] Skipping section 0 - dense segmentation/auto-cropping.")
    #
    # # fname_crop = r"E:/work_dir/case1 Movie_57_crop.tif"  # for unit testing purpose
    # fpath_sb = argv.in3
    # if not argv.intermediate2:
    #     print("[INFO] Starting section 1 - ImageJ Stabilizer:")
    #     # currently, ImageJ asks for output directory even with headless plugin (need to be verified from image.sc)
    #     # just prompt "click cancel" if problem persists.
    #     image_sb, fpath_sb = s1.s1(work_dir=work_path, app_path=app_path, fpath_in=fname_crop, fpath_out=argv.in3, argv=argv)
    # else:
    #     print("[INFO] Skipping section 1 - ImageJ Stabilizer.")
    #
    # print("[INFO] Starting section 2 - CaImAn:")
    # st = time()
    # s2.s2(work_dir=work_path, fpath_in=fpath_sb, fpath_out=None, save=True, log=argv.log)
    # print(f"[INFO] Section 2: caiman finished. Total execution time: {int(time() - st) / 60} m {(time.time() - st) % 60} s")


if __name__ == "__main__":
    start_time = time()
    main()
    print(f"total taken {time() - start_time} seconds")
    sys.exit(0)
