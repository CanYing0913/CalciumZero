# Pipeline Setup Instruction

## 1. Check System Requirement

It has been tested stable on Windows 10 

## 2. Create & Setup a new Python environment from Conda for this project.

Note:
- If you don't have conda installed, You should get started by [here](https://www.anaconda.com/), and then get back to this readme file.
- You can change your environment name by replacing [env_name] with your specified environment name.
- If you encounter any issues regarding installing `pyimagej`, `scpjava`, `xarray`, or any other ImageJ-related dependencies, please refer [here](https://github.com/imagej/pyimagej#readme) for help.
- There are in total of TODO parts of dependencies need to be installed for complete execution of the pipeline: Auto-cropping, ImageJ-Stabilizer, caiman-related, and other general libraries.

1. Create environment using conda and mamba
2. Install Auto-cropping dependencies
3. Install ImageJ-related dependencies
4. Install 

```
conda install mamba -n base -c conda-forge
mamba create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
```