# Microamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y > /dev/null
RUN apt-get install -y wget unzip > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
RUN micromamba install -y -n base -c conda-forge \
        python=3.8\
        opencv  \
        pyimagej  \
        openjdk=8 \
        caiman && \
    micromamba clean --all --yes
ENV JAVA_HOME="/usr/local"
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1  
# Retrieve ImageJ and source code
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip &> /dev/null
RUN unzip fiji-linux64.zip > /dev/null
RUN rm fiji-linux64.zip
RUN wget https://github.com/CanYing0913/CalmAn/raw/master/resource/Image_Stabilizer_Headless.class &> /dev/null
RUN mv Image_Stabilizer_Headless.class /tmp/Fiji.app/plugins/Examples
RUN wget https://raw.githubusercontent.com/CanYing0913/CalmAn/master/src/src_detection.py &> /dev/null
RUN wget https://raw.githubusercontent.com/CanYing0913/CalmAn/master/src/src_stabilizer.py &> /dev/null
RUN wget https://raw.githubusercontent.com/CanYing0913/CalmAn/master/src/src_caiman.py &> /dev/null
RUN wget https://raw.githubusercontent.com/CanYing0913/CalmAn/master/src/src_peak_caller.py &> /dev/null
# Retrieve test file
RUN wget https://github.com/CanYing0913/CalmAn/raw/master/test/test.tif &> /dev/null
RUN wget https://raw.githubusercontent.com/CanYing0913/CalmAn/master/test/test_ij_sa.py &> /dev/null
RUN python /tmp/test_ij_sa.py
# RUN python -c "import imagej, cv2, caiman;ij=imagej.init('/tmp/Fiji.app', mode='headless');print(ij.getVersion());\
#                 imp=ij.IJ.openImage('/tmp/test.tif');print(type(imp));"
# Create output directory
# RUN mkdir result
# RUN ls /tmp/


# ARG CACHEBUST=1 # used for output below

# RUN wget https://github.com/CanYing0913/CalmAn/raw/master/test.tif &> /dev/null
# RUN wget https://github.com/CanYing0913/CalmAn/raw/master/test_ij_sa.py &> /dev/null
# RUN ls -a /tmp/
# ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
# RUN cat /tmp/test_ij_sa.py
# RUN python /tmp/test_ij_sa.py


# FROM continuumio/anaconda3

# RUN conda config --set always_yes yes
# RUN conda update --yes conda
# RUN apt-get update && apt-get install -y gcc g++ libgl1
# RUN apt-get install -y ffmpeg libsm6 libxext6 > /dev/null
# RUN apt-get install -y wget unzip > /dev/null
# # try to install mamba
# RUN conda install mamba -n base -c conda-forge
# RUN mamba --help
# RUN mamba create -n caiman -c conda-forge pyimagej openjdk=8
# # RUN conda create -n caiman -c conda-forge caiman
# RUN /bin/bash -c "source activate caiman && caimanmanager.py install"
# RUN mamba install -c conda-forge opencv seaborn caiman
# # && rm -rf /var/lib/apt/lists/* > /dev/null
# WORKDIR /root
# RUN pwd
# # Retrieve ImageJ and source code
# RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip
# # &> /dev/null
# RUN unzip fiji-linux64.zip > /dev/null
# RUN rm fiji-linux64.zip
# RUN pwd
# RUN wget https://github.com/CanYing0913/CalmAn/raw/master/resource/Image_Stabilizer_Headless.class -P /root
# # &> /dev/null
# RUN ls -a /
# RUN pwd
# RUN ls /root
# RUN mv Image_Stabilizer_Headless.class Fiji.app/plugins/Examples
# RUN conda install -c conda-forge -y opencv pyimagej openjdk=8 seaborn

# RUN mamba --help
# RUN micromamba install -y -n base --channel conda-forge \
#         python=3.8\
#         opencv  \
#         pyimagej  \
#         openjdk=8 \
#         caiman && \
#     micromamba clean --all --yes
# ENV JAVA_HOME="/usr/local"
