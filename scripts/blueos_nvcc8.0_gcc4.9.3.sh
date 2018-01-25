#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
## 
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

rm -rf build_blueos_nvcc8.0_gcc4.9.3 >/dev/null
mkdir build_blueos_nvcc8.0_gcc4.9.3 && cd build_blueos_nvcc8.0_gcc4.9.3

module load cmake/3.7.2

LAGHOS_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${LAGHOS_DIR}/host-configs/blueos/nvcc_gcc_4_9_3.cmake \
  -DMFEM_USE_MPI=On \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tcetmp/packages/cuda-8.0 \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc8.0_gcc4.9.3 \
  -DCUB_DIR=$HOME/workspace/cub \
  "$@" \
  ${LAGHOS_DIR}
