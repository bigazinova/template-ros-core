# parameters
ARG REPO_NAME="template-ros-core"

# ==================================================>
# ==> Do not change this code
ARG ARCH=arm32v7
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}
ARG BASE_IMAGE=dt-core


# define base image
FROM duckietown/${BASE_IMAGE}:${BASE_TAG}

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
ENV NCCL_VERSION 2.8.4
ENV CUDNN_VERSION 8.1.1.33

ENV PYTORCH_VERSION 1.7.0
ENV PYTORCHVISION_VERSION 0.8.0a0+2f40a48

ENV TENSORRT_VERSION 7.1.3.4

ENV PYCUDA_VERSION 2021.1

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

RUN ln -s /usr/local/cuda-10.2 /usr/local/cuda
# define repository path
ARG REPO_NAME
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
WORKDIR "${REPO_PATH}"

# create repo directory
RUN mkdir -p "${REPO_PATH}"

# copy dependencies files only
RUN ls -la
COPY ./dependencies-apt.txt "${REPO_PATH}/"
COPY ./dependencies-py.txt "${REPO_PATH}/"

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install apt dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' dependencies-apt.txt | uniq) \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install -r ${REPO_PATH}/dependencies-py.txt
# copy the source code
COPY . "${REPO_PATH}/"

RUN pwd ${REPO_PATH}/
RUN ls ${REPO_PATH}/

RUN cp -r "${REPO_PATH}/packages/dt-core" "${CATKIN_WS_DIR}/src/"
RUN rm -r "${REPO_PATH}/packages/dt-core"

RUN bash ${REPO_PATH}/install_torch.sh

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# define launch script
ENV LAUNCHFILE "${REPO_PATH}/launch.sh"

# define command
CMD ["bash", "-c", "${LAUNCHFILE}"]
# <== Do not change this code
# <==================================================




# maintainer
LABEL maintainer="Konstantin Chaika (pro100kot14@gmail.com)"
