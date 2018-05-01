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


using namespace RAJA;

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel__
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

using namespace RAJA::statement;

RAJA_INDEX_VALUE(ELEM, "ELEM");
RAJA_INDEX_VALUE(NUM_QD_1D, "NUM_QD_1D");
RAJA_INDEX_VALUE(NUM_THREADS, "NUM_THREADS");
RAJA_INDEX_VALUE(NUM_Q_2D, "NUM_Q_2D");
RAJA_INDEX_VALUE(NUM_MAX, "NUM_MAX");


using Pol1 = RAJA::KernelPolicy<
          CudaKernel<
            For<0, cuda_threadblock_exec<64>, 
            Lambda<0>>
          >
      >;    

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel__
void rMassMultAdd2DNested1(
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements)
  );

  kernel<Pol1>(

    segments,

    [=] RAJA_HOST_DEVICE (ELEM e) 
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
            sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
          }
        }
        for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
          const double q2d = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
          for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
            solOut[ijkN(dx,dy,(int)*e,NUM_DOFS_1D)] += q2d * sol_x[dx];
          }
        }
      }
    }
  );
}



using Pol2 = RAJA::KernelPolicy<
          CudaKernelAsync<
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
         const int NUM_QUAD_1D> kernel__
void rMassMultAdd2DNested2(
                    const int numElements,
                    const double* restrict dofToQuad,
                    const double* restrict dofToQuadD,
                    const double* restrict quadToDof,
                    const double* restrict quadToDofD,
                    const double* restrict oper,
                    const double* restrict solIn,
                    double* restrict solOut) {

  //const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);
  //const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_QD_1D>(0,NUM_QUAD_DOFS_1D)
  );

  using shmemDofQuadMap_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>, decltype(segments)>;
  shmemDofQuadMap_t shmDof2Quad; 
  shmemDofQuadMap_t shmQuad2Dof;

  kernel_param<Pol2>(

    segments, 

    RAJA::make_tuple(
      shmDof2Quad,
      shmQuad2Dof,
      0.0), 

    [=] __device__(ELEM e, NUM_QD_1D qd, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof,  double &)
    {
      shmDof2Quad(qd) = dofToQuad[(int)*qd];
      shmQuad2Dof(qd) = quadToDof[(int)*qd];    
      //int block = blockIdx.x;
      //if(block == 0) printf("shmDof2Quad(%d) = %f, shmQuad2Dof(%d) = %f\n",(int)*qd,shmDof2Quad(qd),(int)*qd,shmQuad2Dof(qd));
    },

    [=] __device__(ELEM e, NUM_QD_1D qd, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof, double &) 
    {
      int thread = threadIdx.x;
      int block = blockIdx.x;
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
          //if(thread == 0 && block == 0) printf("oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)] = %f\n",oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)]);
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


#if 1

using Pol3 = RAJA::KernelPolicy<
          CudaKernel<
            SetShmemWindow<
              For<1, cuda_thread_exec, Lambda<0>>,// NUM_QUAD_DOFS_1D
              For<2, cuda_thread_exec,            // init shmSXY
                For<3, seq_exec, Lambda<1>>
              >  
            >,  
            For<0, cuda_threadblock_exec<64>, 
              Lambda<2>
            >
          >
      >;    

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel__
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
  //const int NUM_MAX_1D = (NUM_QUAD_1D<NUM_DOFS_1D)?NUM_DOFS_1D:NUM_QUAD_1D;

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_QD_1D>(0,NUM_QUAD_DOFS_1D),
    TypedRangeSegment<NUM_THREADS>(0,64),
    TypedRangeSegment<NUM_Q_2D>(0,NUM_QUAD_2D)
  );

  using shmemDofQuadMap_t = ShmemTile<cuda_shmem, double,ArgList<1>, SizeList<NUM_QUAD_DOFS_1D>, decltype(segments)>;
  using shmemSXY_t = ShmemTile<cuda_shmem, double, ArgList<2,3>, SizeList<64,NUM_QUAD_2D>, decltype(segments)>; //shmem for sol_xy for each e in local block
  shmemDofQuadMap_t shmDof2Quad; 
  shmemDofQuadMap_t shmQuad2Dof;
  shmemSXY_t shmSXY;

  //printf("rMassMultAdd2DNested NUM_DOFS_1D = %d  NUM_QUAD_1D = %d\n",NUM_DOFS_1D, NUM_QUAD_1D);
  kernel_param<Pol3>(
    segments, 

    RAJA::make_tuple(shmDof2Quad, shmQuad2Dof, shmSXY, 0.0),


    [=] __device__(ELEM e, NUM_QD_1D qd , NUM_THREADS t, NUM_Q_2D qd2, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof, shmemSXY_t shmSXY, double &)
    {
      shmDof2Quad(qd) = dofToQuad[(int)*qd];
      shmQuad2Dof(qd) = quadToDof[(int)*qd];    
    },

    [=] __device__(ELEM e, NUM_QD_1D qd , NUM_THREADS t, NUM_Q_2D qd2, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof, shmemSXY_t shmSXY, double &)
    {
      shmSXY(t,qd2) = 0.0;
    },

    [=] __device__ (ELEM e, NUM_QD_1D qd , NUM_THREADS t, NUM_Q_2D qd2, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof, shmemSXY_t shmSXY, double &) 
    {
 //     double sol_xy[NUM_QUAD_1D][NUM_QUAD_1D];
 //     for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
 //       for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
 //         sol_xy[qy][qx] = 0.0;
 //       }
 //     }
 //
      int thread = threadIdx.x;
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
            //sol_xy[qy][qx] += d2q * sol_x[qx];
            shmSXY(convertIndex<NUM_THREADS, int>(thread),convertIndex<NUM_Q_2D, int>(ijN(qx,qy,NUM_QUAD_1D))) += d2q * sol_x[qx]; 
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          //sol_xy[qy][qx] *= oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)];
          shmSXY(convertIndex<NUM_THREADS, int>(thread),convertIndex<NUM_Q_2D, int>(ijN(qx,qy,NUM_QUAD_1D))) *= oper[ijkN(qx,qy,(int)*e,NUM_QUAD_1D)]; 
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double sol_x[NUM_DOFS_1D];
        for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
          sol_x[dx] = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          //const double s = sol_xy[qy][qx];
          const double s = shmSXY(convertIndex<NUM_THREADS, int>(thread),convertIndex<NUM_Q_2D, int>(ijN(qx,qy,NUM_QUAD_1D)));
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
         const int NUM_QUAD_1D> kernel__
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


template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> kernel__
void rMassMultAdd3DNested2(
                    const int numElements,
                    const double* dofToQuad,
                    const double* dofToQuadD,
                    const double* quadToDof,
                    const double* quadToDofD,
                    const double* oper,
                    const double* solIn,
                    double* __restrict solOut) {

  //const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_DOFS_1D = (NUM_QUAD_1D * NUM_DOFS_1D);

  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_QD_1D>(0,NUM_QUAD_DOFS_1D)
  );

  using shmemDofQuadMap_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<NUM_QUAD_DOFS_1D>, decltype(segments)>;
  shmemDofQuadMap_t shmDof2Quad; 
  shmemDofQuadMap_t shmQuad2Dof;

  kernel_param<Pol2>(

    segments, 

    RAJA::make_tuple(
      shmDof2Quad,
      shmQuad2Dof,
      0.0), 

    [=] __device__(ELEM e, NUM_QD_1D qd, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof,  double &)
    {
      shmDof2Quad(qd) = dofToQuad[(int)*qd];
      shmQuad2Dof(qd) = quadToDof[(int)*qd];    
    },

    [=] RAJA_HOST_DEVICE(ELEM e, NUM_QD_1D qd, shmemDofQuadMap_t shmDof2Quad, shmemDofQuadMap_t shmQuad2Dof, double &) 
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
            const double s = solIn[ijklN(dx,dy,dz,(int)*e,NUM_DOFS_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              //sol_x[qx] += dofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * s;
              sol_x[qx] += shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qx,dx,NUM_QUAD_1D))) * s;
            }
          }
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            //const double wy = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wy = shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qy,dy,NUM_QUAD_1D)));
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              sol_xy[qy][qx] += wy * sol_x[qx];
            }
          }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
          //const double wz = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
          const double wz = shmDof2Quad(convertIndex<NUM_QD_1D, int>(ijN(qz,dz,NUM_QUAD_1D)));
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
            sol_xyz[qz][qy][qx] *= oper[ijklN(qx,qy,qz,(int)*e,NUM_QUAD_1D)];
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
              //sol_x[dx] += quadToDof[ijN(dx,qx,NUM_DOFS_1D)] * s;
              sol_x[dx] += shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dx,qx,NUM_DOFS_1D))) * s;
            }
          }
          for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
            //const double wy = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
            const double wy = shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dy,qy,NUM_DOFS_1D)));
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
              sol_xy[dy][dx] += wy * sol_x[dx];
            }
          }
        }
        for (int dz = 0; dz < NUM_DOFS_1D; ++dz) {
          //const double wz = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
          const double wz = shmQuad2Dof(convertIndex<NUM_QD_1D, int>(ijN(dz,qz,NUM_DOFS_1D)));
          for (int dy = 0; dy < NUM_DOFS_1D; ++dy) {
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx) {
              solOut[ijklN(dx,dy,dz,(int)*e,NUM_DOFS_1D)] += wz * sol_xy[dy][dx];
            }
          }
        }
      }
    }
  );
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
  const int blck = 256;
  const int grid = (numElements+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
  assert(LOG2(DIM)<=4);
  assert((NUM_QUAD_1D&1)==0);
  assert(LOG2(NUM_DOFS_1D-1)<=8);
  assert(LOG2(NUM_QUAD_1D>>1)<=8);
  const unsigned int id = (DIM<<16)|((NUM_DOFS_1D-1)<<8)|(NUM_QUAD_1D>>1);
  static std::unordered_map<unsigned int, fMassMultAdd> call = {
    // 2D
    {0x20001,&rMassMultAdd2DNested2<1,2>},    {0x20101,&rMassMultAdd2DNested2<2,2>},
    {0x20102,&rMassMultAdd2DNested2<2,4>},    {0x20202,&rMassMultAdd2DNested2<3,4>},
    {0x20203,&rMassMultAdd2DNested2<3,6>},    {0x20303,&rMassMultAdd2DNested2<4,6>},
    {0x20304,&rMassMultAdd2DNested2<4,8>},    {0x20404,&rMassMultAdd2DNested2<5,8>},
    {0x20405,&rMassMultAdd2DNested2<5,10>},   {0x20505,&rMassMultAdd2DNested2<6,10>},
    {0x20506,&rMassMultAdd2DNested2<6,12>},   {0x20606,&rMassMultAdd2DNested2<7,12>},
    {0x20607,&rMassMultAdd2DNested2<7,14>},   {0x20707,&rMassMultAdd2DNested2<8,14>},
    {0x20708,&rMassMultAdd2DNested2<8,16>},   {0x20808,&rMassMultAdd2DNested2<9,16>},
    {0x20809,&rMassMultAdd2DNested2<9,18>},   {0x20909,&rMassMultAdd2DNested2<10,18>},
    {0x2090A,&rMassMultAdd2DNested2<10,20>},  {0x20A0A,&rMassMultAdd2DNested2<11,20>},
    {0x20A0B,&rMassMultAdd2DNested2<11,22>},  {0x20B0B,&rMassMultAdd2DNested2<12,22>},
    {0x20B0C,&rMassMultAdd2DNested2<12,24>},  {0x20C0C,&rMassMultAdd2DNested2<13,24>},
    {0x20C0D,&rMassMultAdd2DNested2<13,26>},  {0x20D0D,&rMassMultAdd2DNested2<14,26>},
    {0x20D0E,&rMassMultAdd2DNested2<14,28>},  {0x20E0E,&rMassMultAdd2DNested2<15,28>},
    {0x20E0F,&rMassMultAdd2DNested2<15,30>},  {0x20F0F,&rMassMultAdd2DNested2<16,30>},
    {0x20F10,&rMassMultAdd2DNested2<16,32>},  {0x21010,&rMassMultAdd2DNested2<17,32>},
    // 3D
    {0x30001,&rMassMultAdd3DNested2<1,2>},    {0x30101,&rMassMultAdd3DNested2<2,2>},
    {0x30102,&rMassMultAdd3DNested2<2,4>},    {0x30202,&rMassMultAdd3DNested2<3,4>},
    {0x30203,&rMassMultAdd3DNested2<3,6>},    {0x30303,&rMassMultAdd3DNested2<4,6>},
    {0x30304,&rMassMultAdd3DNested2<4,8>},    {0x30404,&rMassMultAdd3DNested2<5,8>},
    {0x30405,&rMassMultAdd3DNested2<5,10>},   {0x30505,&rMassMultAdd3DNested2<6,10>},
    {0x30506,&rMassMultAdd3DNested2<6,12>},   {0x30606,&rMassMultAdd3DNested2<7,12>},
    {0x30607,&rMassMultAdd3DNested2<7,14>},   {0x30707,&rMassMultAdd3DNested2<8,14>},
    {0x30708,&rMassMultAdd3DNested2<8,16>},   {0x30808,&rMassMultAdd3DNested2<9,16>},
    {0x30809,&rMassMultAdd3DNested2<9,18>},   {0x30909,&rMassMultAdd3DNested2<10,18>},
    {0x3090A,&rMassMultAdd3DNested2<10,20>},  {0x30A0A,&rMassMultAdd3DNested2<11,20>},
    {0x30A0B,&rMassMultAdd3DNested2<11,22>},  {0x30B0B,&rMassMultAdd3DNested2<12,22>},
    {0x30B0C,&rMassMultAdd3DNested2<12,24>},  {0x30C0C,&rMassMultAdd3DNested2<13,24>},
    {0x30C0D,&rMassMultAdd3DNested2<13,26>},  {0x30D0D,&rMassMultAdd3DNested2<14,26>},
    {0x30D0E,&rMassMultAdd3DNested2<14,28>},  {0x30E0E,&rMassMultAdd3DNested2<15,28>},
    {0x30E0F,&rMassMultAdd3DNested2<15,30>},  {0x30F0F,&rMassMultAdd3DNested2<16,30>},
    {0x30F10,&rMassMultAdd3DNested2<16,32>},  {0x31010,&rMassMultAdd3DNested2<17,32>},
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
