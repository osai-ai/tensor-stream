NAME?=argus-tensor-stream
TORCH_VERSION?=1.4.0
DOCKER_NAME="$(NAME)-$(TORCH_VERSION)"

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all build stop build-whl

all: stop build build-whl

build:
	docker build \
	--build-arg TORCH_VERSION=${TORCH_VERSION} \
	-t $(DOCKER_NAME) .

stop:
	-docker stop $(DOCKER_NAME)
	-docker rm $(DOCKER_NAME)

build-whl:
	docker run --rm -it \
		$(GPUS_OPTION) \
		-v $(shell pwd)/dist:/app/dist \
		--name=$(DOCKER_NAME) \
		$(DOCKER_NAME) \
		python3 setup.py sdist bdist_wheel

run-dev:
	docker run --rm -it \
		$(GPUS_OPTION) \
		--net=host \
		-v $(shell pwd):/app \
		--name=$(DOCKER_NAME) \
		$(DOCKER_NAME) \
		bash
