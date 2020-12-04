FROM ubuntu:16.04

# --------------------------
# install opencv and python
# --------------------------
RUN apt-get update && \
  apt-get install -y build-essential apt-utils \
  cmake git libgtk2.0-dev pkg-config libavcodec-dev \
  libavformat-dev libswscale-dev

RUN  apt-get update && apt-get install -y python-dev python-numpy \
  python python3-pip python-dev libtbb2 libtbb-dev \
  libjpeg-dev libjasper-dev libdc1394-22-dev \
  python-opencv libopencv-dev libav-tools python-pycurl \
  libatlas-base-dev gfortran webp qt5-default libvtk6-dev zlib1g-dev

RUN pip3 install --upgrade pip

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install numpy==1.16.0

RUN cd ~/ &&\
  git clone https://github.com/Itseez/opencv.git &&\
  git clone https://github.com/Itseez/opencv_contrib.git &&\
  cd opencv && mkdir build && cd build && cmake  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
  make -j4 && make install && ldconfig && rm -rf ~/opencv*  # Remove the opencv folders to reduce image size

# Set the appropriate link
RUN ln /dev/null /dev/raw1394

RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

COPY requirements.txt /usr/pose_recognizer/requirements.txt
# COPY requirements.txt ./requirements.txt
WORKDIR /usr/pose_recognizer
RUN pip3 install -r requirements.txt
# COPY tf-openpose ./tf-openpose
COPY tf_pose_estimation ./tf_pose_estimation
COPY . /usr/pose_recognizer
RUN cd tf_pose_estimation && python3 setup.py install && cd ..
# RUN rm -rf tf_pose_estimation

CMD ["python3", "-m", "./poses/server/server_multithreaded"]

