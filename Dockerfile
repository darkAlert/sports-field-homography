FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C TZ=UTC
ENV TERM linux

# install some basic utilities
RUN set -xue ;\
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        ffmpeg \
	    software-properties-common \
        wget \
        zip \
        unzip \
        python3-dev \
        python3-pip \
        python3.8-dev \
        python-yaml \
        awscli \
    ;\
    rm -rf /var/lib/apt/lists/*

# set python3.8 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# install common libs and frameworks
RUN pip3 install --upgrade pip ;\
    pip3 install setuptools

# copy all sports-field sources to the container
WORKDIR /sports/sports-field/
COPY . .

# install boost-court requirements:
RUN pip3 install -r requirements.txt

# run the command
#CMD ["/bin/bash"]
ENTRYPOINT ["/bin/bash"]
