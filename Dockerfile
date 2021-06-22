# Build: docker build -t project_name .
# Run: docker run --gpus all -it --rm project_name

# Build from official Nvidia PyTorch image
# GPU-ready with Apex for mixed-precision support
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

USER root
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt update
RUN apt install -y libopenblas-dev libgl1-mesa-glx build-essential g++ git cmake
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Copy all files
ADD . /workspace/project
WORKDIR /workspace/project

# Create myenv
RUN conda env create -f conda_env_gpu.yaml -n partseg
RUN conda init bash


# Set myenv to default virtual environment
RUN echo "source activate partseg" >> ~/.bashrc
