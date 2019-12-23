FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

WORKDIR /install

ADD launch.sh /code
ADD .gitconfig /root/.gitconfig

RUN apt-get update
RUN apt-get install zsh gnupg2 tmux git wget -y

# Python 3.6
WORKDIR /install
RUN apt-get install zlib1g-dev -y
RUN wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz
RUN tar xf Python-3.6.4.tar.xz
WORKDIR /install/Python-3.6.4
RUN ./configure --enable-optimizations --prefix=/usr
RUN make -j 8
# RUN make -j 8 test
RUN make -j 8 install
RUN ln -s /usr/bin/python3.6 /usr/bin/python
# RUN ln -s /usr/bin/python3.6 /usr/bin/python3
WORKDIR /install

# Setuptools
WORKDIR /install
RUN wget https://github.com/pypa/setuptools/archive/38.2.5.tar.gz
RUN tar xf 38.2.5.tar.gz
WORKDIR /install/setuptools-38.2.5
RUN python3.6 bootstrap.py
RUN python3.6 setup.py install --user
WORKDIR /install

# Cython
WORKDIR /install
RUN wget https://github.com/cython/cython/archive/0.28b1.tar.gz
RUN tar xf 0.28b1.tar.gz
WORKDIR /install/cython-0.28b1
RUN python3.6 setup.py build -j 8 install --user
WORKDIR /install

# Numpy
RUN apt-get install gfortran libopenblas-dev liblapack-dev -y
WORKDIR /install
RUN wget https://github.com/numpy/numpy/releases/download/v1.14.1/numpy-1.14.1.tar.gz
RUN tar xf numpy-1.14.1.tar.gz
WORKDIR /install/numpy-1.14.1
RUN python3.6 setup.py build -j 8 install --user
WORKDIR /install

# Scipy
WORKDIR /install
RUN wget https://github.com/scipy/scipy/releases/download/v1.0.0/scipy-1.0.0.tar.xz
RUN tar xf scipy-1.0.0.tar.xz
WORKDIR /install/scipy-1.0.0
RUN python3.6 setup.py build -j 8 install --user
WORKDIR /install

# Yaml
RUN apt-get install libyaml-dev -y
WORKDIR /install
RUN wget https://github.com/yaml/pyyaml/archive/3.12.tar.gz
RUN tar xf 3.12.tar.gz
WORKDIR /install/pyyaml-3.12/
RUN python3.6 setup.py build -j 8 install --user
WORKDIR /install/

# Pytorch
WORKDIR /install
RUN apt-get install cmake -y
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /install/pytorch
RUN git checkout v0.3.1
RUN python3.6 setup.py build -j 8 install --user
WORKDIR /install


RUN rm -r /install


ENV NAME Pouet