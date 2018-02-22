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
#ifndef LAGHOS_RAJA_MALLOC
#define LAGHOS_RAJA_MALLOC

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/util/Exception.hpp"
#include <signal.h>

// ***************************************************************************
extern bool uvm;
extern bool cuda;

namespace mfem {

// *****************************************************************************
#if 0
template<class T> struct rmalloc{
  // ***************************************************************************
  void* operator new(size_t n) {
    rdbg("+]\033[m");
    if (!rconfig::Get().Cuda()) return ::new T[n];
#ifdef __NVCC__
    void *ptr;
    push(new,Purple);
    if (!rconfig::Get().Uvm()){
      cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T));
    }else{
      cuMemAllocManaged((CUdeviceptr*)&ptr, n*sizeof(T),CU_MEM_ATTACH_GLOBAL);
    }
    pop();
    return ptr;
#else
    // We come here when the user requests a manager,
    // but has compiled the code without NVCC
    assert(false);
#endif // __NVCC__
  }
  
  // ***************************************************************************
  void operator delete(void *ptr) {
    rdbg("-]\033[m");
    if (!rconfig::Get().Cuda()) {
      //assert(ptr!=NULL);
      //assert(ptr!=nullptr);
      //if (ptr)
        ::delete[] static_cast<T*>(ptr);
    }
#ifdef __NVCC__
    else {
      push(delete,Fuchsia);
      cuMemFree((CUdeviceptr)ptr);
      pop();
    }
#endif // __NVCC__
    ptr = nullptr;
  }

};

#endif

#if 1
template<class T> struct rmalloc{

  void* operator new(size_t n) {
    rdbg("+]\033[m");
    auto &rm = umpire::ResourceManager::getInstance();
    umpire::Allocator host_allocator = rm.getAllocator("HOST");
    umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

    if (!rconfig::Get().Cuda()) return static_cast<void*>(host_allocator.allocate(n*sizeof(T)));

#ifdef __NVCC__
    void *ptr;
    push(new,Purple);
    if (!rconfig::Get().Uvm()){
      //cuMemAlloc((CUdeviceptr*)&ptr, n*sizeof(T));
      ptr = static_cast<void*>(device_allocator.allocate(n*sizeof(T)));
    }else{
      cuMemAllocManaged((CUdeviceptr*)&ptr, n*sizeof(T),CU_MEM_ATTACH_GLOBAL);
    }
    pop();
    return ptr;
#else
    // We come here when the user requests a manager,
    // but has compiled the code without NVCC
    assert(false);
#endif // __NVCC__

  }
  // ***************************************************************************
  void operator delete(void *ptr) {
    rdbg("-]\033[m");
    auto &rm = umpire::ResourceManager::getInstance();
    umpire::Allocator host_allocator = rm.getAllocator("HOST");
    umpire::Allocator device_allocator = rm.getAllocator("DEVICE");

    if (!rconfig::Get().Cuda()) {
      if(ptr) {
        try {
          host_allocator.deallocate(ptr);
        }
        catch(...) {  
          ::delete[] static_cast<T*>(ptr);
        }  
      }
      else {
        printf("Requesting delete of host side nil pointer\n");
      }
    }  
#ifdef __NVCC__
    else {
      push(delete,Fuchsia);
      if (!rconfig::Get().Uvm()){
        if(ptr) {
          try {
            device_allocator.deallocate(ptr);
          }
          catch(...) {
            cuMemFree((CUdeviceptr)ptr);
          }
        }
        else{
          printf("Requesting delete of device side nil pointer\n");
        }  
      }
      else {
        cuMemFree((CUdeviceptr)ptr);
      }  
      pop();
    }
#endif // __NVCC__
    ptr = nullptr;
  }
};

#endif
} // mfem

#endif // LAGHOS_RAJA_MALLOC
