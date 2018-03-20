###############################################################################
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738930
#
# All rights reserved.
#
# This file is part of the RAJA Performance Suite.
#
# For details about use and distribution, please read raja-perfsuite/LICENSE.
#
###############################################################################

cmake_minimum_required(VERSION 3.3)

project(Laghos CXX)

#
# Initialize the BLT build system
#

if (LAGHOS_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

set(ENABLE_TESTS Off CACHE BOOL "")

include(blt/SetupBLT.cmake)



set(CMAKE_CXX_STANDARD 11)
set(BLT_CXX_STANDARD 11)

#
# Define RAJA settings...
#


set(ENABLE_TESTS Off CACHE Bool "")
set(ENABLE_EXAMPLES Off CACHE Bool "")
set(ENABLE_DOCUMENTATION Off CACHE Bool "")
set(ENABLE_CALIPER Off CACHE Bool "")

set(ENABLE_TBB Off CACHE Bool "")

set(RAJA_USE_CHRONO On CACHE Bool "")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)
set(RAJA_COHERENCE_BLOCK_SIZE 64)

add_definitions(-D__LAMBDA__=1 -D__TEMPLATES__=1 -D__RAJA__=1)

add_subdirectory(tpl/RAJA)

if(ENABLE_CALIPER)
  set(WITH_SAMPLER Off)
  set(WITH_CALLPATH Off)
  set(WITH_NVPROF On)
  set(WITH_MPI On)
  add_definitions("-DLAGHOS_ENABLE_CALIPER")
  add_subdirectory(tpl/caliper)
  get_property(CALIPER_INCLUDE_DIRS DIRECTORY tpl/caliper PROPERTY INCLUDE_DIRECTORIES)
  MESSAGE( STATUS "CALIPER_INCLUDE_DIRS:           " ${CALIPER_INCLUDE_DIRS})
  include_directories(${CALIPER_INCLUDE_DIRS})
endif()

#    LIBRARIES "umpire;umpire_resource;umpire_strategy;umpire_op;umpire_tpl_cnmem;umpire_tpl_judy;umpire_util")
if(ENABLE_UMPIRE)
  set(ENABLE_UMPIRE off)  
  find_package(umpire REQUIRED)
  set(ENABLE_UMPIRE on)  
  MESSAGE( STATUS "umpire Enabled")
  MESSAGE( STATUS "UMPIRE_INCLUDE_DIRS:           " ${UMPIRE_INCLUDE_DIRS})
  add_definitions("-DLAGHOS_ENABLE_UMPIRE")
  blt_register_library(
    NAME umpire
    INCLUDES ${UMPIRE_INCLUDE_DIRS}
    LIBRARIES "umpire")
endif()  

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/tpl/RAJA/include)
include_directories(${CMAKE_BINARY_DIR}/tpl/RAJA/include)
MESSAGE( STATUS "RAJA_INCLUDE_DIRS:         " ${RAJA_INCLUDE_DIRS} )
MESSAGE( STATUS "RAJA_INCLUDE_DIRS:         " ${CMAKE_SOURCE_DIR}/tpl/RAJA/include )
MESSAGE( STATUS "RAJA_INCLUDE_DIRS:         " ${CMAKE_BINARY_DIR}/tpl/RAJA/include )

add_subdirectory(MFEM) 
include_directories(${CMAKE_SOURCE_DIR}/MFEM/)
MESSAGE( STATUS "MFEM_INCLUDE_DIRS1:         " ${CMAKE_SOURCE_DIR}/MFEM/ )
include_directories(${CMAKE_BINARY_DIR}/MFEM/config)
MESSAGE( STATUS "MFEM_INCLUDE_DIRS2:         " ${CMAKE_BINARY_DIR}/MFEM/config )
include_directories(${HYPRE_INCLUDE_DIRS})
include_directories(${METIS_INCLUDE_DIRS})

set(LAGHOS_VERSION_MAJOR 0)
set(LAGHOS_VERSION_MINOR 1)
set(LAGHOS_VERSION_PATCHLEVEL 0)

set(LAGHOS_DEPENDS RAJA)
list(APPEND LAGHOS_DEPENDS mfem)
if(ENABLE_CALIPER)
list(APPEND LAGHOS_DEPENDS caliper)
endif()

if(ENABLE_UMPIRE)
  list(APPEND LAGHOS_DEPENDS umpire)
endif()  

if (ENABLE_OPENMP)
  list(APPEND LAGHOS_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND LAGHOS_DEPENDS cuda)
   if (ENABLE_CUB)
     list(APPEND LAGHOS_DEPENDS cub)
   endif ()
endif() 

set(LAGHOS_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(LAGHOS_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
  set(CUDA_SEPARABLE_COMPILATION On)
  find_package(CUDA REQUIRED)
  find_library(NVTX_LIBRARY
    NAME libnvToolsExt.so
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

  message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
  add_definitions(-D__NVVP__=1)
  LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib)
  LINK_LIBRARIES(nvToolsExt) 
  LINK_LIBRARIES(cudadevrt)
  LINK_LIBRARIES(cuda)
  #  set(CUDA_HOME /usr/tce/packages/cuda/cuda-9.0.176)
  #include_directories(${CUDA_HOME}/samples/common/inc) 
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc) 
  set(LAGHOS_COMPILER "${CUDA_NVCC_EXECUTABLE}")
  list(APPEND LAGHOS_COMPILER ${CMAKE_CXX_COMPILER})
  set(LAGHOS_COMPILER_OPTIONS "${CUDA_NVCC_FLAGS}")
else()
  set(LAGHOS_COMPILER "${CMAKE_CXX_COMPILER}")
  string(TOUPPER ${CMAKE_BUILD_TYPE} LAGHOS_BUILD_TYPE)
  set(LAGHOS_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${LAGHOS_BUILD_TYPE}}")
  list(APPEND LAGHOS_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})
endif()

#configure_file(${CMAKE_SOURCE_DIR}/src/laghos_config.hpp.in
#  ${CMAKE_CURRENT_BINARY_DIR}/bin/laghos_config.hpp)

# Make sure RAJA flag propagate
set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})

#
#
add_subdirectory(src)
