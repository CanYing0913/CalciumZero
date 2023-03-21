import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need to be fine-tuned.
build_exe_options = {
    "excludes": ['PyQt5'],
    'packages': ['imagej', 'PySimpleGUI', 'caiman', 'ipyparallel', 'skimage', 'seaborn'],
    'include_files': ['config.ini']
}

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="pipeline",
    version="0.1",
    description="Test on build for incomplete GUI application",
    options={"build_exe": build_exe_options},
    executables=[Executable("main_GUI.py", base=base)],
)
