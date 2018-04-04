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

rm -rf build_blueos-gcc-4.9.3 2>/dev/null
mkdir build_blueos-gcc-4.9.3 && cd build_blueos-gcc-4.9.3

module load cmake/3.7.2

LAGHOS_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${LAGHOS_DIR}/host-configs/blueos/gcc_4_9_3.cmake \
  -DENABLE_OPENMP=On \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos-gcc-4.9.3 \
  -DENABLE_UMPIRE=On \
  -Dumpire_DIR=$HOME/workspace/CEED/umpire/install_blueos_nvcc9.0_gcc4.9.3/share/umpire/cmake \
  "$@" \
  ${LAGHOS_DIR}
