# Microamba-docker @ https://github.com/mamba-org/micromamba-docker
FROM mambaorg/micromamba:1.0.0

# Retrieve dependencies
USER root
RUN apt-get update && apt-get upgrade -y
RUN apt-get install apt-utils ffmpeg libsm6 libxext6 -y > /dev/null
RUN apt-get install -y wget unzip git > /dev/null && rm -rf /var/lib/apt/lists/* > /dev/null
RUN micromamba install -y -n base -c conda-forge \
        python=3.8 \
        numpy=1.21 \
        seaborn \
        pyimagej  \
        openjdk=8 \
        pysimplegui \
        caiman && \
    micromamba clean --all --yes
ENV JAVA_HOME="/usr/local"
RUN micromamba install -y -n base --no-channel-priority -c https://marcelotduarte.github.io/packages/conda cx_Freeze
WORKDIR "/tmp"
# Retrieve ImageJ and source code
RUN git clone https://github.com/CanYing0913/CaImAn.git
RUN cp CaImAn/resource/Image_Stabilizer_Headless.class CaImAn/Fiji.app/plugins/Examples
# Create IO Mount directory
RUN mkdir mnt
# Create input directory
RUN mkdir mnt/in
# Create output directory
RUN mkdir mnt/out
# Set MAMVA_DOCKERFILE_ACTIVATE (otherwise python will not be found)
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-c", "import numpy; print(numpy.__version__);"]
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/tmp/CaImAn/main.py", "-wd", "/tmp/mnt/out", \
            "-ijp", "/tmp/CaImAn/Fiji.app"]
