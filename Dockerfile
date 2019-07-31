FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools python3-tk \
    libsm6 libxext6 libxrender1 libasound-dev libportaudio2 &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir numpy==1.16.2

# Install PyTorch
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl &&\
    pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl &&\
    rm -rf ~/.cache/pip

# Install python ML packages
RUN pip3 install --no-cache-dir \
    opencv-python==3.4.2.17 \
    scipy==1.2.1 \
    matplotlib==3.0.3 \
    pandas==0.24.1 \
    jupyter==1.0.0 \
    scikit-learn==0.20.2 \
    scikit-image==0.14.2 \
    librosa==0.6.3 \
    pytorch-argus==0.0.8 \
    sounddevice==0.3.13 \
    onnx==1.5.0 \
    onnxruntime==0.4.0 \
    onnx-coreml==0.4.0

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    git checkout 855808f &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    cd .. && rm -rf apex

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
