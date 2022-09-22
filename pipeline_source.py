import time, os, sys
import numpy as np
import matplotlib.pyplot as plt
from random import randint


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
        return


def examine_segmentation(image_i: np.ndarray, image_o: np.ndarray, idx: int):
    """
    QC function to visualize the denseSegmentation() result within Jupyter Notebook

    Parameters:
        image_i: 3D image prior to segmentation
        image_o: 3D image after segmentation
    """
    assert image_i.shape == image_o.shape and 0 <= idx < image_i.shape[0]
    plt.figure(figsize=(16, 6))
    plt.title("Visualization of Dense Segmentation")
    plt.subplot(1, 2, 1)
    plt.imshow(image_i[idx], cmap='gray')
    plt.title("Before")
    plt.subplot(1, 2, 2)
    plt.imshow(image_o[idx], cmap='gray')
    plt.title("After")


def apply_segmentation(image: np.ndarray, z: int, thresh: int, multi_img=False):
    """
    Apply segmentation based on given threshold on frame z.
    This is just a debugging function to process frame by frame.

    Parameters:
        image: 3D tiff image
        z: slice to apply
        thresh: threshold to perform segmentation
        multi_img: True if apply to every frame
    Returns:
        Segmented image.
    """
    if not multi_img:
        data = image[z, ...].copy()
    else:
        data = image.copy()
    data[data <= thresh] = 0
    return data


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


def find_bb(image: np.ndarray) -> tuple:
    """
    Find coordinates of segmentation bounding box given the segmentation image.

    Parameters:
        image: Segmentation 2D image. Note that the object should be near the center.
    Return:
        tuple: return x, y coordinates for upper-left and bottom-right corners of the bounding box.
    """
    x1 = y1 = x2 = y2 = -1
    h, w = image.shape
    mx, my = h // 2, w // 2
    x1 = x2 = mx
    y1 = y2 = my
    # idxx, idxy = mx, my

    scale = 0.10
    # Make sure the object is basically at the center
    assert image[mx, my] != 0, "Object is not close to center of Image!"
    center_col = image[mx, :].copy()
    center_row = image[:, my].copy()
    while x1 >= 0 and center_row[x1]:
        x1 -= 100
    while x2 < h and center_row[x2]:
        x2 += 100
    while y1 >= 0 and center_col[y1]:
        y1 -= 100
    while y2 < w and center_col[y2]:
        y2 += 100
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise IndexError
    x1 -= 100 if x1 >= 100 else x1
    x2 += 100 if x2 <= w - 100 else x2
    y1 -= 100 if y1 >= 100 else y1
    y2 += 100 if y2 <= h - 100 else y2
    return tuple([x1, y1, x2, y2])


def find_bb_3D(image: np.ndarray, stride: int) -> tuple:
    sliceNum, w, h = image.shape
    x1, y1, x2, y2 = find_bb(image[0, ...])
    i = stride
    while i < sliceNum:
        a, b, c, d = find_bb(image[i, ...])
        x1 = a if a < x1 else x1
        y1 = b if b < y1 else y1
        x2 = c if c > x2 else x2
        y2 = d if d > y2 else y2
        i += stride
    return tuple([x1, y1, x2, y2])
