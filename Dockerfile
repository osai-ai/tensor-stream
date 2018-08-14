FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        cmake \
        bash

COPY . /app
WORKDIR /app
RUN rm -rf build && \
	mkdir -p build

WORKDIR /app/build
RUN cmake .. && make

ENTRYPOINT ["./VideoReader"]