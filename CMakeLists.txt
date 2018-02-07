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

# exclude RAJA make targets from top-level build...
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

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})
MESSAGE( STATUS "RAJA_INCLUDE_DIRS:         " ${RAJA_INCLUDE_DIRS} )

add_subdirectory(MFEM) 
include_directories(${CMAKE_SOURCE_DIR}/MFEM/)
MESSAGE( STATUS "MFEM_INCLUDE_DIRS1:         " ${CMAKE_SOURCE_DIR}/MFEM/ )
include_directories(${CMAKE_BINARY_DIR}/MFEM/config)
MESSAGE( STATUS "MFEM_INCLUDE_DIRS2:         " ${CMAKE_BINARY_DIR}/MFEM/config )
include_directories(${HYPRE_INCLUDE_DIRS})
include_directories(${METIS_INCLUDE_DIRS})

#
# Setup variables to pass to Perf suite
#

set(LAGHOS_VERSION_MAJOR 0)
set(LAGHOS_VERSION_MINOR 1)
set(LAGHOS_VERSION_PATCHLEVEL 0)

set(LAGHOS_DEPENDS RAJA)
list(APPEND LAGHOS_DEPENDS mfem)
if(ENABLE_CALIPER)
list(APPEND LAGHOS_DEPENDS caliper)
endif()

if (ENABLE_OPENMP)
  list(APPEND LAGHOS_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND LAGHOS_DEPENDS cuda)
endif() 

set(LAGHOS_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(LAGHOS_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
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
