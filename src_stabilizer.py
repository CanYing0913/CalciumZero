import imagej
import numpy as np
import tifffile
import os
from scyjava import jimport
from time import time
import matplotlib.pyplot as plt


def prints1(text: str):
    print(f"  *  [S1 - ImageJ stabilizer]: {text}")


def s1(work_dir: dir, app_path, fpath_in, fpath_out=None, argv=None) -> tuple[np.ndarray, str]:
    ij = imagej.init(app_path, mode="interactive")
    prints1(f"ImageJ version {ij.getVersion()}")
    # dataset = ij.io().open(fpath_in)
    # imp = ij.py.to_imageplus(dataset)
    imp = ij.IJ.openImage(fpath_in)

    if fpath_out is None:
        fpath_out = fpath_in[fpath_in.rfind(r"\\") + 1: fpath_in.rfind(".tif")] + "_stabilized.tif"
        fpath_out = os.path.join(work_dir, fpath_out)
    else:
        fpath_out = os.path.join(work_dir, fpath_out)
    prints1(f"Using output name: {fpath_out}")

    Transformation = "Translation" if argv.ij_trans == 0 else "Affine"
    MAX_Pyramid_level = argv.ij_maxpl
    update_coefficient = argv.ij_upco
    MAX_iteration = argv.ij_maxiter
    error_tolerance = argv.ij_errtol
    prints1("Using following parameters:")
    prints1(f"\t\tTransformation: {Transformation};")
    prints1(f"\t\tMAX_Pyramid_level: {MAX_Pyramid_level};")
    prints1(f"\t\tupdate_coefficient: {update_coefficient};")
    prints1(f"\t\tMAX_iteration: {MAX_iteration};")
    prints1(f"\t\terror_tolerance: {error_tolerance};")

    prints1("Starting stabilizer in headless mode...")
    st = time()
    ij.IJ.run(imp, "Image Stabilizer Headless",
              "transformation=" + Transformation + " maximum_pyramid_levels=" + str(MAX_Pyramid_level) +
              " template_update_coefficient=" + str(update_coefficient) + " maximum_iterations=" + str(MAX_iteration) +
              " error_tolerance=" + str(error_tolerance))
    prints1(f"Task finishes. Total of {int((time() - st) // 60)} m {int((time() - st) % 60)} s.")
    ij.IJ.saveAs(imp, "Tiff", fpath_out)
    imp.close()
    return tifffile.imread(fpath_out), fpath_out


def examine_stabilizer(image_i: np.ndarray, image_o: np.ndarray, idx: int):
    """
    QC function to visualize the stabilizer result within Jupyter Notebook

    Parameters:
        image_i: 3D image prior to stabilizer
        image_o: 3D image after stabilizer
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
