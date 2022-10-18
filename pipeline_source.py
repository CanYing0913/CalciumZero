# Note: using this file is deprecated.
import time, os, sys
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import cv2
from math import sqrt


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


def denseSegmentation(image: np.ndarray, debug_mode=False):
    """
    Apply the segmentation based on EVERY frame thresholds.

    Parameters:
        image: 3D image in shape of [N, H, W].
        debug_mode: True if want to retrieve threshold list.
    Returns:
        result: Segmented image of original shape
        th_l: List of thresholds.
    """
    result = np.zeros_like(image)
    th_l = []
    for i in range(len(image)):
        th = find_threshold_v2(image[i]) * 2
        if debug_mode:
            th_l.append(th * 2)
        temp = image[i].copy()
        temp[temp <= th] = 0
        temp[temp > th] = 255
        result[i] = temp
    if debug_mode:
        return result, th_l
    else:
        return result


def examine_segmentation(image_i: np.ndarray, image_o: np.ndarray, idx: int):
    """
    QC function to visualize the denseSegmentation() result within Jupyter Notebook

    Parameters:
        image_i: 3D image prior to segmentation
        image_o: 3D image after segmentation
        idx: index to access
    """
    assert image_i.shape == image_o.shape and 0 <= idx < image_i.shape[0]
    plt.figure(figsize=(16, 6))
    plt.title("Visualization of Dense Segmentation")
    plt.subplot(1, 2, 1)
    plt.imshow(image_i[idx, ...], cmap='gray')
    plt.title("Before")
    plt.subplot(1, 2, 2)
    plt.imshow(image_o[idx, ...], cmap='gray')
    plt.title("After")


def find_threshold(data: np.ndarray) -> int:
    """
    The so-called 'algorithm' of this function: Find subpeaks around 'the' peak

    Parameters:
        data: histogram of pixel value in grayscale image
    Returns:
        thresh: appropriate threshold value to do segmentation.
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
            # maxa = newdata1[a]
            temp = np.zeros(256)
            temp[a + 5:] = newdata1[a + 5:]
            newdata1 = temp
        if np.max(newdata2) >= limit:
            b = np.argmax(newdata2)
            # maxb = newdata2[b]
            temp = np.zeros(256)
            temp[:b - 5] = newdata2[:b - 5]
            newdata2 = temp
    if a < 0 and b < 0:
        raise IndexError
    return max(a, b)


def find_threshold_v2(image):
    T = randint(0, 255)
    T_prime = 300
    while T_prime != T:
        bg = image.copy()
        ob = image.copy()
        bg[bg > T] = 0
        ob[ob <= T] = 0
        m1 = np.mean(ob)
        m2 = np.mean(bg)
        T_prime = T
        T = (m1 + m2) // 2
    return T


def generate_histogram(image: np.ndarray, z: int, plot=False):
    """
        Generate histogram for image frame z. you can choose to plot the histogram.

        Parameters:
            image: 3D image
            z: frame index to find the histogram
            plot: True to plot the histogram, False otherwise
        Returns:
            histogram: The histogram corresponding to the image.
        """
    histogram, bin_edges = np.histogram(image[z, ...], bins=256)
    if plot:
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

    return histogram


def find_bb(image: np.ndarray, debug_mode=False) -> tuple:
    """
    Find coordinates of segmentation bounding box given the segmentation image.

    Parameters:
        image: Segmentation 2D image. Note that the object should be near the center.
    Return:
        tuple: return x, y coordinates for upper-left and bottom-right corners of the bounding box.
    """
    h, w = image.shape
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    s1 = 100
    s2 = h * w * 0.8
    xcnts = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if s1 < area < s2:
            xcnts.append(cnt)
    x1 = y1 = x2 = y2 = -1
    image_t = np.ones_like(image)

    prevDiff = sqrt((0.1 * w) ** 2 + (0.1 * h) ** 2)
    finalxcnt = np.zeros_like(xcnts[0])
    finalxcnts = []
    for xcnt in xcnts:
        xcnt_np = np.array(xcnt)
        xcnt_x = xcnt_np[..., 0]
        xcnt_y = xcnt_np[..., 1]
        mean_x = np.mean(xcnt_x)
        mean_y = np.mean(xcnt_y)
        sum_diff = sqrt((mean_x - w // 2) ** 2 + (mean_y - h // 2) ** 2)
        if sum_diff < prevDiff:
            finalxcnts.append(xcnt)

    prevDist = float('inf')
    for xcnt in finalxcnts:
        x,y = xcnt[0][0]
        dist = sqrt((x-w//2)**2+(y-h//2)**2)
        if dist < prevDist:
            prevDist = dist
            finalxcnt = np.array(xcnt)
            xcnt_x = finalxcnt[..., 0]
            xcnt_y = finalxcnt[..., 1]
            x1 = np.min(xcnt_x)  # -int(h*0.05)
            x2 = np.max(xcnt_x)  # +int(h*0.05)
            y1 = np.min(xcnt_y)  # -int(w*0.05)
            y2 = np.max(xcnt_y)  # +int(w*0.05)
    #
    # prevDiff = sum_diff
    # finalxcnt = xcnt
    # x1 = np.min(xcnt_x)  # -int(h*0.05)
    # x2 = np.max(xcnt_x)  # +int(h*0.05)
    # y1 = np.min(xcnt_y)  # -int(w*0.05)
    # y2 = np.max(xcnt_y)  # +int(w*0.05)

    if debug_mode:
        cv2.drawContours(image_t, finalxcnt, -1, (0, 255, 0), 3)

    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = w - 1 if x1 >= w else x2
    y2 = h - 1 if y1 >= h else y2
    if debug_mode:
        cv2.rectangle(image_t, (x1, y1), (x2, y2), (0, 255, 0), 5)
        plt.figure()
        plt.imshow(image_t, cmap="gray")
        plt.show()
    return tuple([x1, y1, x2, y2])


def find_bb_3D_dense(image: np.ndarray) -> tuple:
    sliceNum, h, w = image.shape
    x1, y1, x2, y2 = find_bb(image[0, ...], True)
    for i in range(1, sliceNum):
        x1c, y1c, x2c, y2c = find_bb(image[i, ...], False)
        x1 = x1c if x1 > x1c else x1
        y1 = y1c if y1 > y1c else y1
        x2 = x2c if x2 < x2c else x2
        y2 = y2c if y2 < y2c else y2

    assert not (x1 == y1 == x2 == y2 == -1)
    return tuple([x1, y1, x2, y2])


def apply_bb_3D(image: np.ndarray, bb: tuple, margin: int) -> np.ndarray:
    x1, y1, x2, y2 = bb
    s, h, w = image.shape
    x1 = x1 - margin if x1 - margin >= 0 else 0
    y1 = y1 - margin if y1 - margin >= 0 else 0
    x2 = x2 + margin if x2 + margin < w else w-1
    y2 = y2 + margin if y2 + margin < h else h-1

    result = image.copy()
    result = result[:, y1:y2, x1:x2]
    return result
