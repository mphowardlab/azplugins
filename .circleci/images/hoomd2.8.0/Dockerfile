FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cmake \
    curl \
    git \
    openssh-client \
    python3 \
    python3-dev \
    python3-numpy \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    zlib1g-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# prevent python from loading paths outside the container
ENV PYTHONPATH=/ignore/pythonpath
RUN sed -i -e 's/ENABLE_USER_SITE = None/ENABLE_USER_SITE = False/g' `python3 -c 'import site; print(site.__file__)'`

# openmpi 3.0 library
RUN curl -sSLO https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2 \
   && echo "f699bff21db0125d8cccfe79518b77641cd83628725a1e1ed3e45633496a82d7  openmpi-3.0.0.tar.bz2" | sha256sum -c - \
   && tar -xjf openmpi-3.0.0.tar.bz2 -C /root \
   && cd /root/openmpi-3.0.0 \
   && ./configure --prefix=/usr \
   && make all install \
   && rm -rf /root/openmpi-3.0.0 \
   && rm /openmpi-3.0.0.tar.bz2

# configure unprivileged ci user that will also receive source code
RUN useradd --create-home --shell /bin/bash ci

# get HOOMD source code and unpack for the ci user (for internal build tests)
RUN curl -sSLO https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.8.0.tar.gz \
    && echo "f3cd733bb9a8932a30d65927897eaff894bf5ddb7a95c507ee47e124a7a4f541  hoomd-v2.8.0.tar.gz" | sha256sum -c - \
    && tar -xzf hoomd-v2.8.0.tar.gz -C /home/ci \
    && mv /home/ci/hoomd-v2.8.0 /home/ci/hoomd \
    && rm -f /home/ci/hoomd/hoomd/example_plugin \
    && chown -R ci:ci /home/ci/hoomd \
    && rm /hoomd-v2.8.0.tar.gz

# hoomd + MD + MPCD in various build configurations
# python3 + nocuda + nompi + double
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=off -DENABLE_MPI=off -DSINGLE_PRECISION=off -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/nocuda/nompi/double \
    && make install \
    && rm -rf /root/build

# python3 + nocuda + nompi + single
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=off -DENABLE_MPI=off -DSINGLE_PRECISION=on -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/nocuda/nompi/single \
    && make install \
    && rm -rf /root/build

# python3 + cuda + nompi + double
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=on -DENABLE_MPI=off -DSINGLE_PRECISION=off -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/cuda/nompi/double \
    && make install \
    && rm -rf /root/build

# python3 + cuda + nompi + single
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=on -DENABLE_MPI=off -DSINGLE_PRECISION=on -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/cuda/nompi/single \
    && make install \
    && rm -rf /root/build

# python3 + nocuda + mpi + double
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=off -DENABLE_MPI=on -DSINGLE_PRECISION=off -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DENABLE_MPI_CUDA=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/nocuda/mpi/double \
    && make install \
    && rm -rf /root/build

# python3 + nocuda + mpi + single
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=off -DENABLE_MPI=on -DSINGLE_PRECISION=on -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DENABLE_MPI_CUDA=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/nocuda/mpi/single \
    && make install \
    && rm -rf /root/build

# python3 + cuda + mpi + double
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=on -DENABLE_MPI=on -DSINGLE_PRECISION=off -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DENABLE_MPI_CUDA=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/cuda/mpi/double \
    && make install \
    && rm -rf /root/build

# python3 + cuda + mpi + single
RUN mkdir -p /root/build \
    && cd /root/build \
    && export CFLAGS="" CXXFLAGS="" \
    && cmake /home/ci/hoomd -DPYTHON_EXECUTABLE="`which python3`" -DENABLE_CUDA=on -DENABLE_MPI=on -DSINGLE_PRECISION=on -DENABLE_TBB=off -DBUILD_CGCMM=off -DBUILD_DEM=off -DBUILD_DEPRECATED=off -DBUILD_HPMC=off -DBUILD_JIT=off -DBUILD_METAL=OFF -DBUILD_TESTING=off -DENABLE_MPI_CUDA=off -DCMAKE_INSTALL_PREFIX=/opt/hoomd/python3/cuda/mpi/single \
    && make install \
    && rm -rf /root/build

# switch to ci user
USER ci:ci
