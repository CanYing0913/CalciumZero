import imagej
import imagej.doctor
import scyjava
from scyjava import jimport
import os, sys
import string, time, code
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt
import pipeline_source


def dump_info(img):
    """A handy function to print details of an image object."""
    name = img.name if hasattr(img, 'name') else None  # xarray
    if name is None and hasattr(img, 'getName'):
        name = img.getName()  # Dataset
    if name is None and hasattr(img, 'getTitle'):
        name = img.getTitle()  # ImagePlus
    print(f"* name: {name or 'N/A'}")
    print(f"* type: {type(img)}")
    print(f"* dtype: {img.dtype if hasattr(img, 'dtype') else 'N/A'}")
    print(f"* shape: {img.shape}")
    print(f"* dims: {img.dims if hasattr(img, 'dims') else 'N/A'}")


def apply_segmentation(image: np.ndarray, z: int, thresh: int):
    """
    Parameters:
        image: 3D tiff image
        z: slice to apply
        thresh: threshold to perform segmentation
    """
    data = image[z, ...]
    data[data <= thresh] = 0
    return data


def plot_org_and_after(image: np.ndarray, z: int, thresh: int):
    temp = image[z, ...].copy()
    temp[temp <= thresh] = 0
    plt.figure()
    plt.suptitle(f'Image at slice {z} with threshold {thresh}')
    plt.subplot(1, 2, 1)
    plt.imshow(image[z, ...], cmap='gray')
    plt.title('original')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(temp[:500, :], cmap='gray')
    plt.title('after')
    plt.colorbar()
    plt.show()


def find_threshold(data: np.ndarray) -> int:
    """
    The so-called 'algorithm' of this function: Find subpeaks around 'the' peak
    """
    a = b = -1
    total, maxv = np.sum(data), np.max(data)
    total = int(total * 0.7)
    limit = maxv // 2
    newdata1 = data.copy()
    newdata2 = data.copy()
    for i in range(3):
        if np.max(newdata1) >= limit:
            a = np.argmax(newdata1)
            maxa = newdata1[a]
            temp = np.zeros(256)
            temp[a + 5:] = newdata1[a + 5:]
            newdata1 = temp
        if np.max(newdata2) >= limit:
            b = np.argmax(newdata2)
            maxb = newdata2[b]
            temp = np.zeros(256)
            temp[:b - 5] = newdata2[:b - 5]
            newdata2 = temp
    if a < 0 and b < 0:
        raise IndexError
    return max(a, b)


def plt_histogram(image, z):
    histogram, bin_edges = np.histogram(image, bins=256)
    x, y = image.shape
    x, y = x // 2, y // 2
    # configure and draw the histogram figure
    plt.figure()
    plt.title(f"at frame {z},Value at ROI: {image[x, y]}")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    # plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    return find_threshold(histogram)


def find_bb(image: np.ndarray) -> tuple:
    x1 = y1 = x2 = y2 = -1
    w, h = image.shape
    mx, my = w // 2, h // 2
    x1 = x2 = mx
    y1 = y2 = my
    idxx, idxy = mx, my

    scale = 0.10
    # Make sure the object is basically at the center
    assert image[mx, my] != 0, "Object is not close to center of Image!"
    center_row = image[mx, :].copy()
    center_col = image[:, my].copy()
    while center_row[x1]:
        x1 -= 100
    while center_row[x2]:
        x2 += 100
    while center_col[y1]:
        y1 -= 100
    while center_col[y2]:
        y2 += 100
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise IndexError
    x1 -= 50 if x1 >= 50 else x1
    x2 += 50 if x2 <= w - 50 else x2
    y1 -= 50 if y1 >= 50 else y1
    y2 += 50 if y2 <= h - 50 else y2
    return tuple([x1, y1, x2, y2])


app_path = r"D:\CanYing\Fiji.app"  # Change this to the directory to local imagej
ij = imagej.init(app_path, mode="interactive")
# ij = imagej.init(mode="interactive")
"""
Ignore the following errors if any when executing this cell:
java.lang.ClassNotFoundException: loci.formats.in.URLReader
java.lang.ClassNotFoundException: loci.formats.in.SlideBook6Reader
java.lang.ClassNotFoundException: loci.formats.in.ScreenReader
java.lang.ClassNotFoundException: loci.formats.in.ZarrReader
"""
# print(f"ImageJ version: {ij.getVersion()}")
# fpath_in = f"E:\case1 Movie_57.tif"
# # fpath_in = f"E:\Case2 Movie_58.tif"
# image = ij.io().open(fpath_in)
# dump_info(image)
# xarray = ij.py.from_java(image)
# image = np.array(xarray)
# for z in [0]:  # , 299, 599, 799, 1199, 1499]:
#     thresh = plt_histogram(image[z, ...], z)
#
#     # plot_org_and_after(image, z, thresh)
#     img_after = apply_segmentation(image, z, thresh)
#     x1, y1, x2, y2 = find_bb(img_after)
#     print(x1, y1, x2, y2)
#     img = cv2.rectangle(img_after, (x1, y1), (x2, y2), color=(255, 0, 0))
#     cv2.imshow('check:', img)
#     cv2.waitKey(0)
# print()
# fpath_out = r"E:/case1 Movie_57_stabilized.tif"
# fpath_out = r"E:/Case2 Movie_58.tif"
fpath_out = r"E:/Case3 Movie_59.tif"
imp = ij.io().open(fpath_out)
image = ij.py.from_java(imp)
image = np.array(image)
plt_histogram(image[0,...], 0)
sys.exit(1)
histogram = pipeline_source.generate_histogram(image, 0)
threshold = pipeline_source.find_threshold(histogram)
print(f"threshold at {threshold}")
img_seg = pipeline_source.apply_segmentation(image, 0, threshold, True)
x1, y1, x2, y2 = pipeline_source.find_bb_3D(img_seg, stride=500)
print(f"cropping original image with upper-left corner at ({x1}, {y1}); lower-right corner at ({x2}, {y2}).")
# image = image.astype(np.float32)
img = cv2.rectangle(image[0, ...], (y1, x1), (y2, x2), color=(255, 0, 0))
cv2.imshow('examine', img)
cv2.waitKey(0)
