# Installation

[//]: # (### Table of Contents)

[//]: # ()
[//]: # (- [Use application]&#40;#use-application&#41;.)

[//]: # (- [Develop locally]&#40;#develop-locally&#41;.)

[//]: # (#### Use application [diabled for now])

[//]: # (To use the application, simply download the latest release from based on your operating system:)

[//]: # ()
[//]: # (|    OS     |                                          Windows                                          |                                        MacOS                                        |                                             Ubuntu                                              |)

[//]: # (|:---------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|)

[//]: # (| Installer | [link]&#40;https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-win64.msi&#41; | [link]&#40;https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1.dmg&#41; | [link]&#40;https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-x86_64.AppImage&#41; |)

[//]: # ()
#### Develop locally

It is recommended to use [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage the environment. To install mamba, follow instructions on its [official website](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

For following commands, replace `<env_name>` with the name of your environment.  
Step0. Clone the repository.
```bash
git clone https://github.com/CanYing0913/CalciumZero.git
cd CalciumZero
```
Step1. Follow instructions to install mamba (usually a mini-forge). After that, activate mamba in your shell or open a mini-forge terminal based on your operating system:
```bash
mamba create -n <env_name> -f envs/cz.yaml  # this command assumes you are in the root directory of the project
```
If the command above does not work, try the following:
```bash
mamba create -n <env_name> -c conda-forge -c anaconda caiman pyimagej openjdk=8
```
Step2. To run the application, activate the environment everytime:
```bash
mamba activate <env_name>
```
Step3. Then run the application with GUI:
```bash
python main.py
```

[//]: # (Options:)

[//]: # (- bdist_msi: build a windows installer)

[//]: # (- bdist_dmg: build a macos installer)

[//]: # (- bdist_appimage: build a linux AppImage)
