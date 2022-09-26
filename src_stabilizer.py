import imagej, scyjava
from scyjava import jimport
from time import time


def s1(app_path, fpath_in, fpath_out, *args):
    ij = imagej.init(app_path, mode="interactive")
    imp = ij.IJ.openVirtual(fpath_in)

    param = False
    if len(args) != 0:
        assert len(args) == 5
        param = True
    if param:
        Transformation = args[0]
        MAX_Pyramid_level = args[1]
        update_coefficient = args[2]
        MAX_iteration = args[3]
        error_tolerance = args[4]
    else:
        print("[INFO] Using default parameters.")
        Transformation = "Translation"  # or "Affine"
        MAX_Pyramid_level = 1
        update_coefficient = 0.90
        MAX_iteration = 200
        error_tolerance = 1E-7

    st = time()
    ij.IJ.run(imp, "Image Stabilizer Headless",
              "transformation=" + Transformation + " maximum_pyramid_levels=" + str(MAX_Pyramid_level) +
              " template_update_coefficient=" + str(update_coefficient) + " maximum_iterations=" + str(MAX_iteration) +
              " error_tolerance=" + str(error_tolerance))
    print(f"Task finishes. Total of {int((time() - st) // 60)} m {int((time() - st) % 60)} s.")
    ij.IJ.saveAs(imp, "Tiff", fpath_out)
    imp.close()
