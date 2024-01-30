# Installation

### Table of Contents

- [Use application](#use-application).
- [Develop locally](#develop-locally).

#### Use application

To use the application, simply download the latest release from based on your operating system:

|    OS     |                                          Windows                                          |                                        MacOS                                        |                                             Ubuntu                                              |
|:---------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Installer | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-win64.msi) | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1.dmg) | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-x86_64.AppImage) |

#### Develop locally

It is recommended to use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage the environment. To install mamba, follow instructions on its [official website](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

The for following commands, replace `<env_name>` with the name of your environment.  
Step1. Follow instructions to install mamba (usually a mini-forge). After that, activate mamba in your shell or open a mini-forge terminal based on your operating system:
```bash
mamba env create -n <env_name> -f envs/cz.yaml  # this command assumes you are in the root directory of the project
```
Step2 [Optional]. If you have no intention to package the application, **skip** the following command:
```bash
mamba install -n <env_name> -y --no-channel-priority -c https://marcelotduarte.github.io/packages/conda cx_Freeze
```
Step3. To run the application, activate the environment everytime:
```bash
mamba activate <env_name>
```
Step4a. Then run the application from GUI:
```bash
python main.py
```
Step4b. To run the application from CLI:
```bash
TODO
```
Step5 [Optional]. To package the application, run the following command (note that you must complete step 2):
```bash
python setup.py [option]
```
Options:
- bdist_msi: build a windows installer
- bdist_dmg: build a macos installer
- bdist_appimage: build a linux AppImage
