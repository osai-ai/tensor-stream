FROM floydhub/pytorch:1.0.0-gpu.cuda9cudnn7-py3.38

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm unzip wget sysstat tmux python-setuptools libtcmalloc-minimal4

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

RUN apt-get -y remove cmake
RUN git clone https://github.com/Kitware/CMake.git && \
    cd CMake && ./bootstrap && make && make install

RUN apt-get -y remove protobuf-compiler
RUN git clone https://github.com/protocolbuffers/protobuf.git && \
    cd protobuf && git submodule update --init --recursive && \
    ./autogen.sh

RUN pip install twine

COPY . /app
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV LD_PRELOAD "/usr/lib/libtcmalloc_minimal.so.4"

#ENTRYPOINT python setup.py install && python examples/sample.py
