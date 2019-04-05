NAME=argus-tensor-stream
CUDA=cu10

.PHONY: all build-docker stop build-whl

all: stop build-docker build-whl

build-docker:
	docker build -t $(NAME) -f docker/Dockerfile_$(CUDA) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

build-whl:
	nvidia-docker run --rm -it \
		-v $(shell pwd)/dist:/app/dist \
		--name=$(NAME) \
		$(NAME) \
		python3 setup.py sdist bdist_wheel

run-bash:
	nvidia-docker run --rm -it \
		--net=host \
		-v $(shell pwd):/app \
		--name=$(NAME) \
		$(NAME) \
		bash
