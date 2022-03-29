FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Build nvidia codec headers
RUN git clone -b sdk/8.2 --single-branch https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&\
    cd nv-codec-headers && make install &&\
    cd .. && rm -rf nv-codec-headers

# Build ffmpeg with nvenc support
RUN git clone --depth 1 -b release/4.2 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
     cd FFmpeg &&\
     mkdir ffmpeg_build && cd ffmpeg_build &&\
     ../configure \
     --enable-cuda \
     --enable-cuvid \
     --enable-libx264 \
     --enable-shared \
     --disable-static \
     --disable-doc \
     --extra-cflags=-I/usr/local/cuda/include \
     --extra-ldflags=-L/usr/local/cuda/lib64 \
     --enable-gpl \
     --extra-libs=-lpthread \
     --nvccflags="-gencode arch=compute_75,code=sm_75" &&\
     make -j$(nproc) && make install && ldconfig &&\
     cd ../.. && rm -rf FFmpeg

RUN pip3 install --no-cache-dir \
    twine==1.13.0 \
    awscli==1.16.194 \
    numpy==1.16.4 \
    cmake==3.18 \
    packaging
ARG TORCH_VERSION
# Install PyTorch
RUN pip3 install --no-cache-dir torch==$TORCH_VERSION --extra-index-url https://download.pytorch.org/whl/cu113
RUN git clone https://github.com/doxygen/doxygen.git &&\
    cd doxygen &&\
    git checkout dc89ac0 &&\
    mkdir build &&\
    cd build &&\
    cmake -G "Unix Makefiles" .. &&\
    make install &&\
    cd ../.. && rm -rf doxygen


COPY . /app
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
