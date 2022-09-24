import time

import numpy as np
import matplotlib.pyplot as plt
from pipeline_source import *
import tifffile

if __name__ == "__main__":
    # test for individual frame cropping
    # ifile = tifffile.imread(r"E:/cropped_result.tif")
    # for i in range(len(ifile)):
    #     curimage = ifile[i]
    #     x1, y1, x2, y2 = find_bb(curimage)
    #     cv2.rectangle(curimage, (x1, y1), (x2, y2), [255,255,255], 10)
    #     plt.figure()
    #     plt.imshow(curimage, cmap="gray")
    #     plt.show()

    # test for dense segmentation
    ifile = tifffile.imread(r"E:/Case2 Movie_58.tif")
    ifile = denseSegmentation(ifile)
    tifffile.imwrite(r"E:/Case2 Movie_58_s.tif", ifile)

    # # test for dense cropping
    ifile = tifffile.imread(r"E:/Case2 Movie_58_s.tif")
    st = time.time()
    x1, y1, x2, y2 = find_bb_3D_dense(ifile)
    print(f"finish in {time.time()-st} seconds.")

    infile = tifffile.imread(r"E:/Case2 Movie_58.tif")
    result = apply_bb_3D(infile, (x1, y1, x2, y2), 200)
    tifffile.imwrite(r"E:/Case2 Movie_58_c.tif", result)

    print()
