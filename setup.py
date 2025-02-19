from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but they might need fine-tuning
build_exe_options = {
    "build_exe": "dist",
    "includes": [
        "caiman",
    ],
    "packages": [
        "z5py",
    ],
    "include_files": [
        "config.json",
        "log"
    ],

}
build_msi_options = {}
build_mac_options = {}
build_appimage_options = {}

executables = [
    Executable(
        "main.py",
        copyright="Copyright (C) 2024 CalciumZero",
        base="console",
        # icon="icon.ico",
        target_name="CalciumZero",
        shortcut_name="CalciumZero",
        shortcut_dir="Program",
    )
]

setup(
    name="CalciumZero",
    version="0.5",
    description="CalciumZero Toolbox",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": build_msi_options,
        "bdist_mac": build_mac_options,
        "bdist_appimage": build_appimage_options,
    },
    executables=executables,
)