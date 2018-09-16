FROM floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.31

COPY . /app
WORKDIR /app

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm unzip wget sysstat tmux python-setuptools

RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&\
    cd nv-codec-headers && make install && cd ..

RUN git clone --depth 1 -b release/4.0 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
    cd FFmpeg &&\
    mkdir ffmpeg_build && cd ffmpeg_build &&\
    ../configure \
    --enable-cuda \
    --enable-cuvid \
    --enable-shared \
    --disable-static \
    --disable-programs \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-gpl \
    --extra-libs=-lpthread \
    --nvccflags="-gencode arch=compute_61,code=sm_61 -O3" &&\
    make -j$(nproc) && make install && ldconfig