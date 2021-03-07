FROM ubuntu:18.04
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN apt-get update -y
 
RUN apt-add-repository -y "ppa:ubuntu-toolchain-r/test"
 
RUN apt-get install -yq --no-install-suggests --no-install-recommends \
   build-essential \
   git valgrind \
   binutils build-essential libtool texinfo \
   gzip zip unzip patchutils curl git \
   make cmake ninja-build automake bison flex gperf \
   grep sed gawk python bc \
   zlib1g-dev libexpat1-dev libmpc-dev \
   libglib2.0-dev libfdt-dev libpixman-1-dev wget
 
RUN git clone --depth=50 --branch=master https://github.com/vortexgpgpu/vortex vortexgpgpu/vortex
WORKDIR vortexgpgpu/vortex
 
RUN git submodule update --init --recursive
 
ENV CXX=${CXX:-g++}
ENV CXX_FOR_BUILD=${CXX_FOR_BUILD:-g++}
ENV CC=${CC:-gcc}
ENV CC_FOR_BUILD=${CC_FOR_BUILD:-gcc}
RUN gcc --version
 
RUN ci/toolchain_install.sh -all
 
ENV RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
ENV VERILATOR_ROOT=/opt/verilator
ENV PATH=$VERILATOR_ROOT/bin:$PATH
RUN make -s
RUN ./ci/test_runtime.sh
