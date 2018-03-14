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
#include "../raja.hpp"
#include "RAJA/RAJA.hpp"

using RAJA::Index_type;
using RAJA::View;
using RAJA::Layout;
using RAJA::nested::Lambda;
using RAJA::nested::ArgList;


// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rMassMultAdd2D(
#ifndef __TEMPLATES__
                    const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
#endif
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
  }
#ifdef __LAMBDA__
  );
#endif
}
using namespace RAJA;
using namespace RAJA::nested;

using Pol1 = nested::Policy<
          CudaKernel<
            For<0, cuda_threadblock_exec<64>, 
            Lambda<0>>
          >
      >;    

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rMassMultAdd2DNested1(
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {

  //forall(e,numElements,
  //printf("rMassMultAdd2DNested NUM_DOFS_1D = %d  NUM_QUAD_1D = %d\n",NUM_DOFS_1D, NUM_QUAD_1D);
  nested::forall<Pol1>(
    RAJA::make_tuple(RangeSegment(0,numElements)),
    [=] __device__(int e) 
    {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
    }
  );
}


RAJA_INDEX_VALUE(ELEM, "ELEM");
RAJA_INDEX_VALUE(NUM_QD_1D, "NUM_QD_1D");

using Pol2 = nested::Policy<
          CudaKernel<
            SetShmemWindow<
              For<1, cuda_thread_exec, Lambda<0>>// NUM_QUAD_DOFS_1D
            >,  
            For<0, cuda_threadblock_exec<64>, 
              Lambda<1>
            >
          >
      >;    

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rMassMultAdd2DNested2(
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {

  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_QD_1D>(0,NUM_QUAD_DOFS_1D)
  );

  using shmemDofQuadMap_t = SharedMemory<cuda_shmem, double,NUM_QUAD_DOFS_1D>;
  ShmemWindowView<shmemDofQuadMap_t,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>,decltype(segments)> shmDof2Quad; 
  ShmemWindowView<shmemDofQuadMap_t,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>,decltype(segments)> shmQuad2Dof;

  //printf("rMassMultAdd2DNested NUM_DOFS_1D = %d  NUM_QUAD_1D = %d\n",NUM_DOFS_1D, NUM_QUAD_1D);
  nested::forall<Pol2>(
    segments, 

    [=] __device__(ELEM e, NUM_QD_1D qd)
    {
      shmDof2Quad(qd) = dofToQuad[(int)*qd];
      shmQuad2Dof(qd) = quadToDof[(int)*qd];    
    },

    [=] RAJA_HOST_DEVICE(ELEM e, NUM_QD_1D qd) 
    {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,(int)*e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            //sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
            sol_x[qx] += shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qx,dx,NUM_QUAD_1D))) * s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          //const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double d2q = shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qy,dy,NUM_QUAD_1D)));
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            //sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
            sol_x[dx] += shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dx,qx,NUM_DOFS_1D))) * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          //const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          const double q2d = shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dy,qy,NUM_DOFS_1D)));
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,(int)*e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
    }
  );
}


#if 0

using Pol3 = nested::Policy<
          CudaKernel<
            SetShmemWindow<
              For<1, cuda_thread_exec, Lambda<0>>,// NUM_QUAD_DOFS_1D
              For<2, cuda_threadblock_exec<64>>
            >,  
            For<0, cuda_threadblock_exec<64>, 
              Lambda<1>
            >
          >
      >;    

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
void rMassMultAdd2DNested3(
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {

  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_QD_1D>(0,NUM_QUAD_DOFS_1D),
    TypedRangeSegment<NUM_THREADS>(0,64),
    TypedRangeSegment<NUM_Q_2D>(0,NUM_QUAD_2D)
  );

  using shmemDofQuadMap_t = SharedMemory<cuda_shmem, double,NUM_QUAD_DOFS_1D>;
  using shmemSXY_t = SharedMemory<cuda_shmem, double, NUM_QUAD_2D * 64>; //shmem for sol_xy for each e in local block
  ShmemWindowView<shmemDofQuadMap_t,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>,decltype(segments)> shmDof2Quad; 
  ShmemWindowView<shmemDofQuadMap_t,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>,decltype(segments)> shmQuad2Dof;
  ShmemWindowView<shmemSXY_t,Arglist<2,3>,SizeList<64,NUM_QUAD_2D>,decltype(segments)> shmSXY;

  //printf("rMassMultAdd2DNested NUM_DOFS_1D = %d  NUM_QUAD_1D = %d\n",NUM_DOFS_1D, NUM_QUAD_1D);
  nested::forall<Pol2>(
    segments, 

    [=] __device__(ELEM e, NUM_QD_1D qd , NUM_THREADS t, NUM_Q_2D)
    {
      shmDof2Quad(qd) = dofToQuad[(int)*qd];
      shmQuad2Dof(qd) = quadToDof[(int)*qd];    
    },

    [=] __device__(ELEM e, NUM_QD_1D qd , NUM_THREADS t, NUM_Q_2D)
    {
      shmSXY = 0.0;
    }


    [=] RAJA_HOST_DEVICE(ELEM e, NUM_QD_1D qd, NUM_THREADS t, NUM_Q_2D)) 
    {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0.0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          sol_x[qy] = 0.0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijkN(dx,dy,(int)*e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            //sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)]* s;
            sol_x[qx] += shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qx,dx,NUM_QUAD_1D))) * s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          //const double d2q = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double d2q = shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qy,dy,NUM_QUAD_1D)));
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += d2q * sol_x[qx];
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] *= oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)];
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xy[qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            //sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
            sol_x[dx] += shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dx,qx,NUM_DOFS_1D))) * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          //const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          const double q2d = shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dy,qy,NUM_DOFS_1D)));
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,(int)*e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
    }
  );
}

#endif

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel
#endif
void rMassMultAdd3D(
#ifndef __TEMPLATES__
                    const int NUM_DOFS_1D,
                    const int NUM_QUAD_1D,
#endif
                    const int numElements,
                    const double* dofToQuad,
                    const double* dofToQuadD,
                    const double* quadToDof,
                    const double* quadToDofD,
                    const double* oper,
                    const double* solIn,
                    double* __restrict solOut) {
#ifndef __LAMBDA__
  const int e = blockDim.x * blockIdx.x + threadIdx.x;
  if (e < numElements)
#else
  forall(e,numElements,
#endif
  {
    double sol_xyz[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D];
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] = 0;
        }
      }
    }
    for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
      double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xy[qy][qx] = 0;
        }
      }
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        double sol_x[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_x[qx] = 0;
        }
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xy[qy][qx] += wy * sol_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
          }
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,e,NUM_QUAD_1D)];
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double sol_xy[NUM_DOFS_1D][NUM_DOFS_1D];
      for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_xy[dy][dx] = 0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double s = sol_xyz[qz][qy][qx];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            sol_xy[dy][dx] += wy * sol_x[dx];
          }
        }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
        const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
          }
        }
      }
    }
  }
#ifdef __LAMBDA__
  );
#endif
}

// *****************************************************************************
typedef void (*fMassMultAdd)(const int numElements,
                             const double* dofToQuad,
                             const double* dofToQuadD,
                             const double* quadToDof,
                             const double* quadToDofD,
                             const double* oper,
                             const double* solIn,
                             double* __restrict solOut);

// *****************************************************************************
void rMassMultAdd(const int DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* dofToQuad,
                  const double* dofToQuadD,
                  const double* quadToDof,
                  const double* quadToDofD,
                  const double* op,
                  const double* x,
                  double* __restrict y) {
  push(Lime);
#ifndef __LAMBDA__
  const int grid = numElements;
  const int blck = NUM_QUAD_1D;
#endif
#ifdef __TEMPLATES__
  const unsigned int id = (DIM<<16)|(NUM_DOFS_1D<<8)|(NUM_QUAD_1D);
  assert(LOG2(DIM)<=8);
  assert(LOG2(NUM_DOFS_1D)<=8);
  assert(LOG2(NUM_QUAD_1D)<=8);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    // 2D
    {0x20102,&rMassMultAdd2D<1,2>},
    {0x20202,&rMassMultAdd2D<2,2>},
    {0x20203,&rMassMultAdd2D<2,3>},
    {0x20204,&rMassMultAdd2D<2,4>},
    {0x20205,&rMassMultAdd2D<2,5>},
    {0x20206,&rMassMultAdd2D<2,6>},
    {0x20207,&rMassMultAdd2D<2,7>},
    {0x20208,&rMassMultAdd2D<2,8>},
    
    {0x20302,&rMassMultAdd2D<3,2>},
    {0x20303,&rMassMultAdd2D<3,3>},
    {0x20304,&rMassMultAdd2D<3,4>},
    {0x20305,&rMassMultAdd2D<3,5>},
    {0x20306,&rMassMultAdd2DNested2<3,6>}, /* temp test of nested */
    {0x20307,&rMassMultAdd2D<3,7>},
    {0x20308,&rMassMultAdd2D<3,8>},
    
    {0x20402,&rMassMultAdd2D<4,2>},
    {0x20403,&rMassMultAdd2D<4,3>},
    {0x20404,&rMassMultAdd2D<4,4>},
    {0x20405,&rMassMultAdd2D<4,5>},
    {0x20406,&rMassMultAdd2DNested2<4,6>},/* temp test of nested */
    {0x20407,&rMassMultAdd2D<4,7>},
    {0x20408,&rMassMultAdd2D<4,8>},

    {0x20502,&rMassMultAdd2D<5,2>},
    {0x20503,&rMassMultAdd2D<5,3>},
    {0x20504,&rMassMultAdd2D<5,4>},
    {0x20505,&rMassMultAdd2D<5,5>},
    {0x20506,&rMassMultAdd2D<5,6>},
    {0x20507,&rMassMultAdd2D<5,7>},
    {0x20508,&rMassMultAdd2D<5,8>},
    {0x20509,&rMassMultAdd2D<5,9>},
    {0x2050A,&rMassMultAdd2D<5,10>},

    {0x20602,&rMassMultAdd2D<6,2>},
    {0x20603,&rMassMultAdd2D<6,3>},
    {0x20604,&rMassMultAdd2D<6,4>},
    {0x20605,&rMassMultAdd2D<6,5>},
    {0x20606,&rMassMultAdd2D<6,6>},
    {0x20607,&rMassMultAdd2D<6,7>},
    {0x20608,&rMassMultAdd2D<6,8>},
    {0x20609,&rMassMultAdd2D<6,9>},
    {0x2060A,&rMassMultAdd2D<6,10>},
    {0x2060C,&rMassMultAdd2D<6,12>},
    
    {0x20702,&rMassMultAdd2D<7,2>},
    {0x20703,&rMassMultAdd2D<7,3>},
    {0x20704,&rMassMultAdd2D<7,4>},
    {0x20705,&rMassMultAdd2D<7,5>},
    {0x20706,&rMassMultAdd2D<7,6>},
    {0x20707,&rMassMultAdd2D<7,7>},
    {0x20708,&rMassMultAdd2D<7,8>},
    {0x20709,&rMassMultAdd2D<7,9>},
    {0x2070A,&rMassMultAdd2D<7,10>},
    {0x2070C,&rMassMultAdd2D<7,12>},

    // 3D
    {0x30202,&rMassMultAdd3D<2,2>},
    {0x30203,&rMassMultAdd3D<2,3>},
    {0x30204,&rMassMultAdd3D<2,4>},
    {0x30205,&rMassMultAdd3D<2,5>},
    {0x30206,&rMassMultAdd3D<2,6>},
    {0x30207,&rMassMultAdd3D<2,7>},
    {0x30208,&rMassMultAdd3D<2,8>},
    {0x30209,&rMassMultAdd3D<2,9>},
    
    {0x30302,&rMassMultAdd3D<3,2>},
    {0x30303,&rMassMultAdd3D<3,3>},
    {0x30304,&rMassMultAdd3D<3,4>},
    {0x30305,&rMassMultAdd3D<3,5>},
    {0x30306,&rMassMultAdd3D<3,6>},
    {0x30307,&rMassMultAdd3D<3,7>},
    {0x30308,&rMassMultAdd3D<3,8>},
    
    {0x30402,&rMassMultAdd3D<4,2>},
    {0x30403,&rMassMultAdd3D<4,3>},
    {0x30404,&rMassMultAdd3D<4,4>},
    {0x30405,&rMassMultAdd3D<4,5>},
    {0x30406,&rMassMultAdd3D<4,6>},
    {0x30407,&rMassMultAdd3D<4,7>},
    {0x30408,&rMassMultAdd3D<4,8>},
  };
  if(!call[id]){
    printf("\n[rMassMultAdd] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rMassMultAdd,id,grid,blck,
        numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#else
  if (DIM==1) assert(false);
  if (DIM==2)
    call0(rMassMultAdd2D,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
  if (DIM==3)
    call0(rMassMultAdd3D,id,grid,blck,
          NUM_DOFS_1D,NUM_QUAD_1D,
          numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,op,x,y);
#endif
  pop();
}
