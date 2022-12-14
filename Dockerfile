# FROM mambaorg/micromamba:1.0.0
# # FROM ubuntu:20.04

# USER root
# # Retrieve basic functionalities
# RUN apt-get update
# RUN apt-get upgrade -y
# RUN apt-get install ffmpeg libsm6 libxext6  -y > /dev/null
# RUN apt-get install -y wget unzip > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
# # Retrieve ImageJ and source code
# RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip &> /dev/null
# RUN unzip fiji-linux64.zip > /dev/null
# RUN rm fiji-linux64.zip
# RUN wget https://github.com/CanYing0913/CalmAn/blob/master/Image_Stabilizer_Headless.class &> /dev/null
# RUN mv Image_Stabilizer_Headless.class Fiji.app/plugins/Examples
# RUN micromamba install -y -n base --channel conda-forge \
#         python=3.8\
#         opencv  \
#         pyimagej  \
#         openjdk=8 \
#         caiman && \
#     micromamba clean --all --yes
# ENV JAVA_HOME="/usr/local"

# ARG CACHEBUST=1 # used for output below

# RUN wget https://github.com/CanYing0913/CalmAn/raw/master/test.tif &> /dev/null
# RUN wget https://github.com/CanYing0913/CalmAn/raw/master/test_ij_sa.py &> /dev/null
# RUN ls -a /tmp/
# ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
# RUN cat /tmp/test_ij_sa.py
# RUN python /tmp/test_ij_sa.py


FROM continuumio/anaconda3

RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN apt-get update && apt-get install -y gcc g++ libgl1
RUN conda create -n caiman -c conda-forge caiman
RUN /bin/bash -c "source activate caiman && caimanmanager.py install"
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y wget unzip > /dev/null 
# && rm -rf /var/lib/apt/lists/* > /dev/null
WORKDIR /root
RUN pwd
# Retrieve ImageJ and source code
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
# &> /dev/null
RUN unzip fiji-linux64.zip > /dev/null
RUN rm fiji-linux64.zip
RUN pwd
RUN wget https://github.com/CanYing0913/CalmAn/blob/master/Image_Stabilizer_Headless.class -P /root
# &> /dev/null
RUN ls -a /
RUN pwd
RUN ls /root
RUN mv Image_Stabilizer_Headless.class Fiji.app/plugins/Examples
RUN conda install -y opencv pyimagej openjdk=8
# RUN micromamba install -y -n base --channel conda-forge \
#         python=3.8\
#         opencv  \
#         pyimagej  \
#         openjdk=8 \
#         caiman && \
#     micromamba clean --all --yes
# ENV JAVA_HOME="/usr/local"
