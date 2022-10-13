import imagej, scyjava
import numpy as np
import tifffile
import os
from scyjava import jimport
from time import time


def prints1(text: str):
    print(f"  *  [S1 - ImageJ stabilizer]: {text}")


def s1(work_dir: dir, app_path, fpath_in, fpath_out=None, argv=None) -> tuple[np.ndarray, str]:
    ij = imagej.init(app_path, mode="interactive")
    prints1(f"ImageJ version {ij.getVersion()}")
    # dataset = ij.io().open(fpath_in)
    # imp = ij.py.to_imageplus(dataset)
    imp = ij.IJ.openImage(fpath_in)

    if fpath_out is None:
        fpath_out = fpath_in[fpath_in.rfind("/") + 1: fpath_in.rfind(".tif")] + "_stabilized.tif"
        fpath_out = os.path.join(work_dir, fpath_out)
    else:
        fpath_out = os.path.join(work_dir, fpath_out)
    prints1(f"Using output name: {fpath_out}")

    prints1("Using default parameters.")
    Transformation = "Translation" if argv.TODO else "Affine"
    MAX_Pyramid_level = argv.TODO
    update_coefficient = argv.TODO
    MAX_iteration = argv.TODO
    error_tolerance = argv.TODO

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
