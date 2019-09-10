NAME?=argus-tensor-stream
CUDA?=cu10
DOCKER_NAME="$(NAME)-$(CUDA)"

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all build-docker stop build-whl

all: stop build-docker build-whl

build-docker:
	docker build -t $(DOCKER_NAME) -f docker/Dockerfile_$(CUDA) .

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

run-bash:
	docker run --rm -it \
	    $(GPUS_OPTION) \
		--net=host \
		-v $(shell pwd):/app \
		--name=$(DOCKER_NAME) \
		$(DOCKER_NAME) \
		bash
