import imagej
from time import time


def prints1(txt):
    print(txt)


ij = imagej.init('/tmp/Fiji.app', mode='interactive')
imp = ij.IJ.openImage('/temp/test.tif')
Transformation = "Translation"
MAX_Pyramid_level = 1.0
update_coefficient = 0.9
MAX_iteration = 200
error_tolerance = 1E-7
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
