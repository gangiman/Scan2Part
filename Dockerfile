# Build: docker build -t project_name .
# Run: docker run --gpus all -it --rm project_name
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

ADD . /workspace/project
WORKDIR /workspace/project

RUN conda env create -f conda_env_gpu.yaml -n partseg
RUN conda init bash

RUN echo "source activate partseg" >> ~/.bashrc

RUN conda install openblas-devel -c anaconda

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
    -v --no-deps \
    --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --install-option="--blas=openblas"
