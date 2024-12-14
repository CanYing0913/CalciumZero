# CalciumZero:  A user-friendly prep processing pipeline for fluorescence calcium imaging
 - Features
 - [Run](#configure-environment)
## Running the pipeline:  
- You can use our pre-built GUI application. Download based on your platform [here](#running-locally-through-our-distribution).
- If you encounter problems for our GUI application, please submit an issue.
- You can directly run it on Google Colab. See [Colab Instructions](#part-i-running-on-colab) to run it within Colab.  
- You can run it using our Docker image.
- **[Discouraged]** You can manually install all the dependencies to manually run it and further develop on it. See [Instructions on local](#part-iii-running-locally) for a detailed explanation.  
## Configure environment
If you just want to interact and use this work, the **best** way is to launch our provided application package across platorm.  

|    OS     |                                          Windows                                          |                                        MacOS                                        |                                             Ubuntu                                              |
|:---------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| Installer | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-win64.msi) | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1.dmg) | [link](https://github.com/CanYing0913/CaImAn/blob/distribution/CalciumZero-0.1-x86_64.AppImage) |
If you want to develop this application further, you can refer to [Instructions on local](#running-locally-) to run it locally.
## Running on Colab
You will need to follow the link [here](https://colab.research.google.com/drive/1BvHYZRoOla47MwVeV5_0H2-Vko1nm9yW?usp=sharing) to our Colab notebook. Note that our Colab notebook is lightweight, free to go. Prior to run the pipeline on Colab, you should have your input files located in your Google Drive. At the beginning of the notebook, we will ask you for permissions to mount your Google Drive on Google Colab runtime.  
## Running on Docker  
Docker Image is already on DockerHub. Current version: `0.1`.  
- First make sure you have [Docker](https://www.docker.com/) installed on your computer.  
- Pull the docker image using command `docker image pull canying0913/caiman_pipeline:0.1`.  
- Download our provided [launch script](https://raw.githubusercontent.com/CanYing0913/CaImAn/master/run_pipeline.py). Use python to run this script, as it will launch a container with the image we just pulled. If you need help with using the launcher script, you can run `python3 run_pipeline.py -h` to have a detailed explanation on its parameters.
# Citation:z
Xiaofu He*, Yutong Gao, Yian Wang, Xuchen Wang, Qifan Jiang, Bin Xu*,Imaging Analysis of Calcium Activities in Brain Organoid Model of Neuropsychiatric Disorder, Brain Informactics 2024 short paper.
