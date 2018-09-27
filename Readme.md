### Steps to build Docker container and run sample:
docker build -t videoreader . \
nvidia-docker run -ti videoreader bash \
python setup_lin.py install \
python Sample.py