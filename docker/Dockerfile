# MIT License
#
# Copyright (C) 2025 Goodarz Mehr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# UniTR Docker Configuration Script
#
# This script performs all the necessary steps for creating a UniTR Docker
# image.
#
# The base Docker image is Ubuntu 20.04 with CUDA 11.3. If you want to use a
# different base image, you may need to modify "ubuntu2004/x86_64" when
# fetching keys, based on your Ubuntu release and system architecture.

# Build Arguments (Case Sensitive):
#
# USER: username inside each container, set to "untr" by default.

# Installation:
#
# 1. Install Docker on your system (https://docs.docker.com/engine/install/).
# 2. Install the Nvidia Container Toolkit
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
# It exposes your Nvidia graphics card to Docker containers.
# 3. Install the Nvidia Container Runtime and set it as the default runtime
# (https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).
# 4. In the Dockerfile directory, run
#
# docker build --no-cache --rm --build-arg ARG -t unitr:develop .

# Usage:
#
# Launch a container by running
#
# docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
# -v [path/to/UniTR]:/home/unitr
# -v [path/to/dataset]:/dataset
# --shm-size 32g -it unitr:develop /bin/bash
#
# Use "nvidia-smi" to ensure your graphics card is visible inside the
# container. Then, as the root user inside the "unitr" directory, run
#
# python setup.py develop

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Define build arguments and environment variables.

ARG USER=untr

ENV USER=${USER}
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere"

# Add new user and install prerequisite packages.

WORKDIR /home

RUN useradd -m ${USER}

RUN set -xue && apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
&& apt update \
&& apt install -y build-essential cmake debhelper git wget libsdl2-2.0-0 libsm6 libgl1-mesa-glx libomp5 pip unzip \
libjpeg8 libtiff5 software-properties-common nano fontconfig g++ gcc libglib2.0-0 openmpi-bin openmpi-common gpustat \
libopenmpi-dev libgtk2.0-dev libnvidia-gl-550 libnvidia-common-550 python-is-python3

RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 SharedArray==3.0.0 \
nuscenes-devkit==1.0.5 numba==0.48.0 numpy==1.23.5 shapely==1.8.4 networkx==3.1 llvmlite==0.31.0 av2==0.2.1 tqdm \
tensorboardX easydict pyyaml scikit-image pyquaternion spconv-cu113 kornia opencv-python \
-f https://download.pytorch.org/whl/torch_stable.html

RUN FORCE_CUDA=1 pip install --no-cache-dir torch-scatter

RUN FORCE_CUDA=1 pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

USER ${USER}
