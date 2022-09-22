import numpy as np
import matplotlib.pyplot as plt
from pipeline_source import find_threshold_v2
import tifffile

if __name__ == "__main__":
    # for i in range(92):
    #     temp = np.load(r"E:/roi_"+str(i)+".npy")
    #     plt.figure()
    #     plt.imshow(temp)
    #     plt.show()
    #     print(temp.shape)
    ifile = tifffile.imread(r"E:/resized_case1.tif")
    ofile = np.zeros_like(ifile)
    th_l = []
    for i in range(len(ifile)):
        th = find_threshold_v2(None, ifile[i])*2
        th_l.append(th*2)
        temp = ifile[i].copy()
        temp[temp <= th] = 0
        ofile[i] = temp
    tifffile.imwrite(r"E:/segmented_case1.tif", ofile)

    print()