"""
Source file for Section 0 - Dense Segmentation and object detection
"""
from math import sqrt
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile


def prints0(text: str):
    print(f"  *  [S0]: {text}")


def s0(work_dir: str, fname_in: str, margin=200, fname_out=None, save=True, debug=False):
    image_i = tifffile.imread(fname_in)
    image_o = denseSegmentation(image_i, debug)
    if save:
        if fname_out is None:
            fname_out = fname_in[fname_in.rfind("/") + 1: fname_in.rfind(".tif")] + "_out.tif"
            fname_out = os.path.join(work_dir, fname_out)
        prints0(f"Using output name: {fname_out}")
        tifffile.imwrite(fname_out, image_o)

    x1, y1, x2, y2 = find_bb_3D_dense(image_o, debug)
    image_o = apply_bb_3D(image_i, (x1, y1, x2, y2), margin)
    return image_o, fname_out


def denseSegmentation(image: np.ndarray, debug_mode=False):
    """
    Apply the segmentation based on EVERY frame thresholds.

    Parameters:
        image: 3D image in shape of [N, H, W].
        debug_mode: True if you want to retrieve threshold list.
    Returns:
        result: Segmented image of original shape
        th_l: List of thresholds.
    """
    result = np.zeros_like(image)
    th_l = []
    for i in range(len(image)):
        th = find_threshold(image[i]) * 2
        if debug_mode:
            th_l.append(th * 2)
        temp = image[i].copy()
        temp[temp <= th] = 0
        temp[temp > th] = 255
        result[i] = temp
    return result, th_l


def find_threshold(image) -> int:
    """
    Find threshold to perform segmentation for a time slice.

    Parameters:
        image: Frame/time slice to find the threshold
    Returns:
        threshold: threshold value to perform segmentation
    """
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
        x, y = xcnt[0][0]
        dist = sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
        if dist < prevDist:
            prevDist = dist
            finalxcnt = np.array(xcnt)
            xcnt_x = finalxcnt[..., 0]
            xcnt_y = finalxcnt[..., 1]
            x1 = np.min(xcnt_x)
            x2 = np.max(xcnt_x)
            y1 = np.min(xcnt_y)
            y2 = np.max(xcnt_y)

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


def find_bb_3D_dense(image: np.ndarray, debug_mode=False) -> tuple:
    sliceNum, h, w = image.shape
    x1, y1, x2, y2 = find_bb(image[0, ...], debug_mode)
    for i in range(1, sliceNum):
        x1c, y1c, x2c, y2c = find_bb(image[i, ...], debug_mode)
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
    x2 = x2 + margin if x2 + margin < w else w - 1
    y2 = y2 + margin if y2 + margin < h else h - 1

    result = image.copy()
    result = result[:, y1:y2, x1:x2]
    return result
