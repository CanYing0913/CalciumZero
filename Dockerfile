# Microamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update && apt-get upgrade -y
RUN apt-get install apt-utils ffmpeg libsm6 libxext6 -y > /dev/null
RUN apt-get install -y wget unzip git > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
RUN micromamba install -y -n base -c conda-forge \
        python=3.9 \
        numpy=1.23 \
        seaborn \
        opencv  \
        pyimagej  \
        openjdk=8 \
        caiman && \
    micromamba clean --all --yes
# RUN micromamba update --all
ENV JAVA_HOME="/usr/local"
# Retrieve ImageJ and source code
RUN ls
RUN git clone https://github.com/CanYing0913/CaImAn.git
WORKDIR "/tmp/CaImAn"
RUN wget https://downloads.imagej.net/fiji/latest/fiji-linux64.zip &> /dev/null
RUN unzip fiji-linux64.zip > /dev/null
RUN rm fiji-linux64.zip
RUN cp /tmp/CaImAn/resource/Image_Stabilizer_Headless.class /tmp/CaImAn/Fiji.app/plugins/Examples
# Retrieve test file
RUN mkdir /tmp/input
# Create output directory
RUN mkdir /tmp/result
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-c", "import numpy; print(numpy.__version__);"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/tmp/CaImAn/main.py", "-wd", "/tmp/result", \
            "-ijp", "/tmp/Fiji.app"]
