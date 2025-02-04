"""
Source file for Section 1 - ImageJ Stabilizer
Last edited on July 17, 2024
Copyright @ Yian Wang (canying0913@gmail.com) - 2024
"""
import imagej
from pathlib import Path
from typing import Union, List, Dict


def run_plugin(
        ijp: Union[Path, str],
        filename: str,
        work_dir: Union[Path, str],
        s1_params: Union[List, Dict]
) -> Path:
    def remove_suffix(input_string, suffix):
        if suffix and input_string.endswith(suffix):
            return input_string[:-len(suffix)]
        return input_string

    ij = imagej.init(str(ijp), mode='headless')
    fname_out = Path(filename).stem + '_stab.tif'
    fname_out = Path(work_dir).joinpath(fname_out)
    imp = ij.IJ.openImage(filename)
    if type(imp) is None:
        raise TypeError(f'imp failed to initialize with path {filename}')
    if isinstance(s1_params, list):
        Transformation = "Translation" if s1_params[0] == 0 else "Affine"
        MAX_Pyramid_level = s1_params[1]
        update_coefficient = s1_params[2]
        MAX_iteration = s1_params[3]
        error_tolerance = s1_params[4]
    elif isinstance(s1_params, dict):
        Transformation = s1_params['Transformation']
        MAX_Pyramid_level = s1_params['MAX_Pyramid_level']
        update_coefficient = s1_params['update_coefficient']
        MAX_iteration = s1_params['MAX_iteration']
        error_tolerance = s1_params['error_tolerance']
    else:
        raise TypeError(f'Unknown type {type(s1_params)} for s1_params.')
    # f(f"Using output name {fname_out} for {fname}. Starting...")
    ij.IJ.run(imp, "Image Stabilizer Headless",
              "transformation=" + Transformation + " maximum_pyramid_levels=" + str(MAX_Pyramid_level) +
              " template_update_coefficient=" + str(update_coefficient) + " maximum_iterations=" +
              str(MAX_iteration) + " error_tolerance=" + str(error_tolerance))
    ij.IJ.saveAs(imp, "Tiff", str(fname_out))
    imp.close()
    # f(f"{fname_out} exec finished.")
    return fname_out


def print_param(ij_params, f):
    Transformation = "Translation" if ij_params[0] == 0 else "Affine"
    MAX_Pyramid_level = ij_params[1]
    update_coefficient = ij_params[2]
    MAX_iteration = ij_params[3]
    error_tolerance = ij_params[4]
    f("Using following parameters:")
    f(f"Transformation: {Transformation};")
    f(f"MAX_Pyramid_level: {MAX_Pyramid_level};")
    f(f"update_coefficient: {update_coefficient};")
    f(f"MAX_iteration: {MAX_iteration};")
    f(f"error_tolerance: {error_tolerance};")
