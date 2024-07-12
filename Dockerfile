# Microamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update && apt-get upgrade -y
RUN apt-get install apt-utils ffmpeg libsm6 libxext6 -y > /dev/null
RUN apt-get install -y wget unzip git > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
# Retrieve ImageJ and source code
WORKDIR "/tmp"
RUN git clone https://github.com/CanYing0913/CalciumZero.git
RUN cp CaImAn/resource/Image_Stabilizer_Headless.class CaImAn/Fiji.app/plugins/Examples
RUN micromamba install -y -n base -f CalciumZero/envs/cz.yaml && \
    micromamba clean --all --yes
ENV JAVA_HOME="/usr/local"
# Create IO Mount directory
RUN mkdir mnt
# Create input directory
RUN mkdir mnt/in
# Create output directory
RUN mkdir mnt/out
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-c", "import numpy; print(numpy.__version__);"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/tmp/CaImAn/main.py"]
