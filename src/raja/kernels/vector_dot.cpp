// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "raja.hpp"
#include <cuda.h>
#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include <cub/cub.cuh>

using namespace RAJA;

#if 0


double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  //ReduceDecl(Sum,dot,0.0);
  ReduceSum<cuda_reduce_atomic<256>,double> dot(0.0);
  //cout << "Setting Grid Stride Mode to occupancy_size" << endl; 
  static bool s_GSModeSet = false;
  if(!s_GSModeSet) { 
    RAJA::policy::cuda::getGridStrideMode() = RAJA::policy::cuda::GridStrideMode::occupancy_size;
    s_GSModeSet = true;
  }  
  forall<cuda_exec<256> >(0,N, [=] __device__(int i) {
    dot += vec1[i] * vec2[i];
  });
  return dot.get();
}

#endif

#if 1
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {
  ReduceDecl(Sum,dot,0.0);
  forall(i,N,dot += vec1[i] * vec2[i];);
  return dot.get();
}

#endif

#if 0
double vector_dot(const int N,
                  const double* __restrict vec1,
                  const double* __restrict vec2) {

  static bool s_cubAllocFlag = false;
  static size_t temp_storage_bytes=0;
  static double *d_in;
  static double *d_out;
  static void *temp_storage;

  if(! s_cubAllocFlag) {
    cudaMallocManaged(&d_in,1024 * 1024 * sizeof(double));
    cudaMallocManaged(&d_out, 1 * sizeof(double));
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_in, d_out, 1024 * 1024);
    cudaMalloc(&temp_storage,temp_storage_bytes);
    //cudaDeviceSynchronize();
    s_cubAllocFlag = true;
  }

  //for(int i=0;i<128*1024;i++) d_in[i] = 0.0;
  for(int i=0;i<N;i++)
    d_in[i] = vec1[i] * vec2[i];

  //cudaDeviceSynchronize();
  cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_in, d_out, N);
  cudaDeviceSynchronize();
  return d_out[0];
}

#endif

