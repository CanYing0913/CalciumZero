import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need to be fine-tuned.
build_exe_options = {
    "excludes": ['PyQt5'],
    'packages': ['imagej', 'caiman', 'ipyparallel', 'skimage', 'seaborn'],
    'include_files': ['cache', 'Fiji.app'],
    'silent_level': '1'
}

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable(
        'main.py',
        base=base,
        target_name='CalciumZero'
    )
]

setup(
    name="CalciumZero",
    version="0.4",
    description="Test on build for incomplete GUI application",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": {},
        "bdist_dmg": {},
        "bdist_appimage": {},
    },
    executables=executables,
)
