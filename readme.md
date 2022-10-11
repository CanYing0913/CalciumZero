# Pipeline Setup Instruction  
## 1. Check System Requirement  
It has been tested stable on Windows 10 

## 2. Create & Setup a new Python environment from Conda for this project.  
Note:  
- If you don't have conda installed, You should get started by [here](https://www.anaconda.com/), and then get back to this readme file.
- You can change your environment name by replacing `[env_name]` with your specified environment name when executing following commands.
- If you encounter any issues regarding installing `pyimagej`, `scpjava`, `xarray`, or any other ImageJ-related dependencies, please refer [here](https://github.com/imagej/pyimagej#readme) for help.
- There are in total of `#TODO` parts of dependencies need to be installed for complete execution of the pipeline: Auto-cropping, ImageJ-Stabilizer, caiman-related, and other general libraries.

1. Create environment using conda and mamba
2. Install Auto-cropping dependencies
3. Install ImageJ-related dependencies
4. Install 

```
conda install mamba -n base -c conda-forge
mamba create -n pyimagej -c conda-forge pyimagej openjdk=8
conda activate pyimagej
```

## 3. Install Fiji ImageJ to local and add headless stabilizer plugin to ImageJ
Up until latest change @ `2022.10.11`, there is no docker for this project. Please install fiji ImageJ [here](https://imagej.net/software/fiji/downloads).  
After successfully installing Fiji ImageJ, copy the compiled plugin file, `Image_Stabilizer.java`, under ImageJ folder: `fiji.app/plugins/`

## 4. Before running the pipeline  
There are 2 ways for you to run the pipeline. An interactive way is to leverage Jupyter Notebook to customize IO, parameters, etc. Another "headless" way is to use Python script. While there is no difference between them, using Jupyter Notebook better visualize the intermediate and results for better quality control purposes.  
#### 4.a Running in Jupyter Notebook  
To run the pipeline in Jupyter Notebook, open Anaconda Prompt, and type the following commands to activate your created environment and open notebook:
```
conda activate [env_name]
jupyter notebook
```
Navigate to the ipynb file of the pipeline, `main.ipynb` and open it.  
#### 4.b Running in python console or IDEs  
To run the pipeline as a script, you can directly use the script, `main.py` as a command-line script, or run it within any IDE (e.g. [Pycharm](https://www.jetbrains.com/pycharm/))  
## 5. Running the pipeline  
For both notebook and script, you should either modify the following parameters or pass parameters as arguments to the script:  
```
work_dir: str, the path to a folder where most of results get stored. (not used in Jupyter Notebook)
app_path: str, the path to local installation of Fiji ImageJ, fiji.app
#TODO

```