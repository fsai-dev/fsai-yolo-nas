FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Volta - p3, Ampere - p4, Turing - g4
ARG TORCH_CUDA_ARCH_LIST="Voltal;Turing"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MKL_THREADING_LAYER GNU
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    wget \
    curl \
    ca-certificates curl ffmpeg libsm6 libxext6 \
    git wget ninja-build protobuf-compiler libprotobuf-dev build-essential python3-opencv tmux nvidia-container-toolkit cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Set Python 3.9 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN ln -sv /usr/bin/python3 /usr/bin/python


# Install the required python packages globally
ENV PATH="$PATH:/root/.local/bin"

# Set the current working directory
WORKDIR /home

RUN \
    # Upgrade pip
    pip install --upgrade pip && \
    # Create the input and output directories
    mkdir -p data && \
    mkdir -p yolo-nas-output

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

RUN git clone https://github.com/Deci-AI/super-gradients.git super-gradients
RUN pip install --user -e super-gradients

RUN pip install clearml pycocotools

WORKDIR /home/super-gradients
