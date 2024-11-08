FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 9.0 9.0a"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MKL_THREADING_LAYER="GNU"
ENV DEBIAN_FRONTEND="noninteractive"

ENV NO_ALBUMENTATIONS_UPDATE="1"

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
    nvidia-container-toolkit \
    git wget ninja-build protobuf-compiler libprotobuf-dev build-essential python3-opencv tmux vim cmake \
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

RUN pip install --upgrade pip

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

RUN git clone --depth 1 https://github.com/Deci-AI/super-gradients.git super-gradients
RUN pip install --user -e super-gradients

WORKDIR /home/super-gradients


RUN pip install comet_ml pycocotools albumentations==1.4.18


