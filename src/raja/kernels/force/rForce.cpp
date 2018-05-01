
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
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMult2D(
#ifndef __TEMPLATES__
                         const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
#endif
                         const int numElements,
                         const double* restrict L2DofToQuad,
                         const double* restrict H1QuadToDof,
                         const double* restrict H1QuadToDofD,
                         const double* restrict stressJinvT,
                         const double* restrict e,
                         double* restrict v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double e_xy[NUM_QUAD_2D];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      e_xy[i] = 0;
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      double e_x[NUM_QUAD_1D];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        e_x[qy] = 0;
      }
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        const double r_e = e[ijkN(dx,dy,el,L2_DOFS_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
        }
      }
    }
    for (int c = 0; c < 2; ++c) {
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] = 0.0;
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double Dxy[H1_DOFS_1D];
        double xy[H1_DOFS_1D];
        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          Dxy[dx] = 0.0;
          xy[dx]  = 0.0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
          const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
            xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
          const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
          }
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
RAJA_INDEX_VALUE(NUM_L2DQ, "NUM_QDL2D");
RAJA_INDEX_VALUE(NUM_H1QD, "NUM_H1DNQD");
// for the transpose variant
RAJA_INDEX_VALUE(NUM_L2QD, "NUM_L2QD");
RAJA_INDEX_VALUE(NUM_H1DQ, "NUM_H1DQ");
using Pol1 = RAJA::KernelPolicy<
          CudaKernelAsync<
            SetShmemWindow<
              For<1, cuda_thread_exec, Lambda<0>>
            >,  
            SetShmemWindow<
              For<2, cuda_thread_exec, Lambda<1>>
            >,  
            For<0, cuda_threadblock_exec<32>, 
              Lambda<2>
            >
          >
      >;    

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMult2DNested1(
#ifndef __TEMPLATES__
                         const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
#endif
                         const int numElements,
                         const double* restrict L2DofToQuad,
                         const double* restrict H1QuadToDof,
                         const double* restrict H1QuadToDofD,
                         const double* restrict stressJinvT,
                         const double* restrict e,
                         double* restrict v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  
  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_L2DQ>(0,NUM_QUAD_1D * L2_DOFS_1D),
    TypedRangeSegment<NUM_H1QD>(0,H1_DOFS_1D * NUM_QUAD_1D)
  );

  using shmemL2DofToQuad_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<NUM_QUAD_1D * L2_DOFS_1D>, decltype(segments)>;
  using shmemH1QuadToDof_t = ShmemTile<cuda_shmem, double,ArgList<2>,SizeList<H1_DOFS_1D * NUM_QUAD_1D>, decltype(segments)>;
  shmemL2DofToQuad_t shmL2DofToQuad; 
  shmemH1QuadToDof_t shmH1QuadToDof;
  shmemH1QuadToDof_t shmH1QuadToDofD;

  kernel_param<Pol1>(

    segments, 

    RAJA::make_tuple(
      shmL2DofToQuad,
      shmH1QuadToDof,
      shmH1QuadToDofD,
      0.0), 

    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
    {
      shmL2DofToQuad(dq) = L2DofToQuad[(int)*dq];
    },


    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
    {
      shmH1QuadToDof(qd) = H1QuadToDof[(int)*qd];
      shmH1QuadToDofD(qd) = H1QuadToDofD[(int)*qd];
    },


    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
    {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          e_x[qy] = 0;
        }
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          const double r_e = e[ijkN(dx,dy,(int)*el,L2_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            //e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
            e_x[qx] += shmL2DofToQuad(convertIndex<NUM_L2DQ,int>(ijN(qx,dx,NUM_QUAD_1D))) * r_e;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          //const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wy = shmL2DofToQuad(convertIndex<NUM_L2DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
          }
        }
      }
      for (int c = 0; c < 2; ++c) {
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklNM(c,dx,dy,(int)*el,NUM_DOFS_1D,numElements)] = 0.0;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dxy[H1_DOFS_1D];
          double xy[H1_DOFS_1D];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dxy[dx] = 0.0;
            xy[dx]  = 0.0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double esx = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,(int)*el,NUM_DIM,NUM_QUAD_1D)];
            const double esy = e_xy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,(int)*el,NUM_DIM,NUM_QUAD_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              //Dxy[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
              //xy[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              Dxy[dx] += esx * shmH1QuadToDofD(convertIndex<NUM_H1QD,int>(ijN(dx,qx,H1_DOFS_1D)));
              xy[dx]  += esy * shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dx,qx,H1_DOFS_1D)));
            }
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            //const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            //const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            const double wy  = shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dy,qy,H1_DOFS_1D)));
            const double wDy = shmH1QuadToDofD(convertIndex<NUM_H1QD,int>(ijN(dy,qy,H1_DOFS_1D)));
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              v[_ijklNM(c,dx,dy,(int)*el,NUM_DOFS_1D,numElements)] += wy* Dxy[dx] + wDy*xy[dx];
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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMultTranspose2D(
#ifndef __TEMPLATES__
                                  const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
#endif
                                  const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double vStress[NUM_QUAD_2D];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      double v_Dxy[NUM_QUAD_2D];
      double v_xDy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        v_Dxy[i] = v_xDy[i] = 0;
      }
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        double v_x[NUM_QUAD_1D];
        double v_Dx[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          v_x[qx] = v_Dx[qx] = 0;
        }

        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          const double r_v = v[_ijklNM(c,dx,dy,el,NUM_DOFS_1D,numElements)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
            v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
            ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]) +
             (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,el,NUM_DIM,NUM_QUAD_1D)]));
        }
      }
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e[ijkN(dx,dy,el,L2_DOFS_1D)] = 0;
      }
    }
    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      double e_x[L2_DOFS_1D];
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e_x[dx] = 0;
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
        }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijkN(dx,dy,el,L2_DOFS_1D)] += e_x[dx] * w;
        }
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}

#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMultTranspose2DNested1(
#ifndef __TEMPLATES__
                                  const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
#endif
                                  const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  
  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_L2QD>(0,NUM_QUAD_1D * L2_DOFS_1D),
    TypedRangeSegment<NUM_H1DQ>(0,H1_DOFS_1D * NUM_QUAD_1D)
  );

  using shmemL2QuadToDof_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<NUM_QUAD_1D * L2_DOFS_1D>, decltype(segments)>;
  using shmemH1DofToQuad_t = ShmemTile<cuda_shmem, double,ArgList<2>,SizeList<H1_DOFS_1D * NUM_QUAD_1D>, decltype(segments)>;
  shmemL2QuadToDof_t shmL2QuadToDof; 
  shmemH1DofToQuad_t shmH1DofToQuad;
  shmemH1DofToQuad_t shmH1DofToQuadD;

  kernel_param<Pol1>(

    segments, 

    RAJA::make_tuple(
      shmL2QuadToDof,
      shmH1DofToQuad,
      shmH1DofToQuadD,
      0.0), 

    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
    {
      shmL2QuadToDof(qd) = L2QuadToDof[(int)*qd];
    },


    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
    {
      shmH1DofToQuad(dq) = H1DofToQuad[(int)*dq];
      shmH1DofToQuadD(dq) = H1DofToQuadD[(int)*dq];
    },


    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
  {
    double vStress[NUM_QUAD_2D];
    for (int i = 0; i < NUM_QUAD_2D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      double v_Dxy[NUM_QUAD_2D];
      double v_xDy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        v_Dxy[i] = v_xDy[i] = 0;
      }
      for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
        double v_x[NUM_QUAD_1D];
        double v_Dx[NUM_QUAD_1D];
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          v_x[qx] = v_Dx[qx] = 0;
        }

        for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
          const double r_v = v[_ijklNM(c,dx,dy,(int)*el,NUM_DOFS_1D,numElements)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            //v_x[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            //v_Dx[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
            v_x[qx]  += r_v * shmH1DofToQuad(convertIndex<NUM_H1DQ,int>(ijN(qx,dx,NUM_QUAD_1D)));
            v_Dx[qx] += r_v * shmH1DofToQuadD(convertIndex<NUM_H1DQ,int>(ijN(qx,dx,NUM_QUAD_1D)));

          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          //const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          //const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
          const double wy  = shmH1DofToQuad(convertIndex<NUM_H1DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
          const double wDy = shmH1DofToQuadD(convertIndex<NUM_H1DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] += v_Dx[qx] * wy;
            v_xDy[ijN(qx,qy,NUM_QUAD_1D)] += v_x[qx]  * wDy;
          }
        }
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          vStress[ijN(qx,qy,NUM_QUAD_1D)] +=
            ((v_Dxy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(0,c,qx,qy,(int)*el,NUM_DIM,NUM_QUAD_1D)]) +
             (v_xDy[ijN(qx,qy,NUM_QUAD_1D)] * stressJinvT[ijklmNM(1,c,qx,qy,(int)*el,NUM_DIM,NUM_QUAD_1D)]));
        }
      }
    }
    for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e[ijkN(dx,dy,(int)*el,L2_DOFS_1D)] = 0;
      }
    }
    for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
      double e_x[L2_DOFS_1D];
      for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
        e_x[dx] = 0;
      }
      for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
        const double r_v = vStress[ijN(qx,qy,NUM_QUAD_1D)];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          //e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
          e_x[dx] += r_v * shmL2QuadToDof(convertIndex<NUM_L2QD ,int>(ijN(dx,qx,L2_DOFS_1D)));
        }
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        //const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
        const double w = shmL2QuadToDof(convertIndex<NUM_L2QD ,int>(ijN(dy,qy,L2_DOFS_1D)));
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijkN(dx,dy,(int)*el,L2_DOFS_1D)] += e_x[dx] * w;
        }
      }
    }
  }
#ifdef __LAMBDA__
         );
#endif
}

// *****************************************************************************
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
void rForceMult3D(
#ifndef __TEMPLATES__
                  const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
#endif
                  const int numElements,
                  const double* restrict L2DofToQuad,
                  const double* restrict H1QuadToDof,
                  const double* restrict H1QuadToDofD,
                  const double* restrict stressJinvT,
                  const double* restrict e,
                  double* restrict v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double e_xyz[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      e_xyz[i] = 0;
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          e_x[qy] = 0;
        }
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          const double r_e = e[ijklN(dx,dy,dz,el,L2_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        const double wz = L2DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] += wz * e_xy[ijN(qx,qy,NUM_QUAD_1D)];
          }
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] = 0;
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        double Dxy_x[H1_DOFS_1D * H1_DOFS_1D];
        double xDy_y[H1_DOFS_1D * H1_DOFS_1D];
        double xy_z[H1_DOFS_1D * H1_DOFS_1D] ;
        for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d) {
          Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dx_x[H1_DOFS_1D];
          double x_y[H1_DOFS_1D];
          double x_z[H1_DOFS_1D];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            const double esx = r_e * stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            const double esy = r_e * stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            const double esz = r_e * stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
              x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
            }
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dxy_x[ijN(dx,dy,H1_DOFS_1D)] += Dx_x[dx] * wy;
              xDy_y[ijN(dx,dy,H1_DOFS_1D)] += x_y[dx]  * wDy;
              xy_z[ijN(dx,dy,H1_DOFS_1D)]  += x_z[dx]  * wy;
            }
          }
        }
        for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
          const double wz  = H1QuadToDof[ijN(dz,qz,H1_DOFS_1D)];
          const double wDz = H1QuadToDofD[ijN(dz,qz,H1_DOFS_1D)];
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)] +=
                ((Dxy_x[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xDy_y[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xy_z[ijN(dx,dy,H1_DOFS_1D)]  * wDz));
            }
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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
void rForceMult3DNested1(
#ifndef __TEMPLATES__
                  const int NUM_DIM,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int L2_DOFS_1D,
                  const int H1_DOFS_1D,
#endif
                  const int numElements,
                  const double* restrict L2DofToQuad,
                  const double* restrict H1QuadToDof,
                  const double* restrict H1QuadToDofD,
                  const double* restrict stressJinvT,
                  const double* restrict e,
                  double* restrict v) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
  
  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_L2DQ>(0,NUM_QUAD_1D * L2_DOFS_1D),
    TypedRangeSegment<NUM_H1QD>(0,H1_DOFS_1D * NUM_QUAD_1D)
  );

  using shmemL2DofToQuad_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<NUM_QUAD_1D * L2_DOFS_1D>, decltype(segments)>;
  using shmemH1QuadToDof_t = ShmemTile<cuda_shmem, double,ArgList<2>,SizeList<H1_DOFS_1D * NUM_QUAD_1D>, decltype(segments)>;
  shmemL2DofToQuad_t shmL2DofToQuad; 
  shmemH1QuadToDof_t shmH1QuadToDof;
  shmemH1QuadToDof_t shmH1QuadToDofD;

  kernel_param<Pol1>(

    segments, 

    RAJA::make_tuple(
      shmL2DofToQuad,
      shmH1QuadToDof,
      shmH1QuadToDofD,
      0.0), 

    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
    {
      shmL2DofToQuad(dq) = L2DofToQuad[(int)*dq];
    },


    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
    {
      shmH1QuadToDof(qd) = H1QuadToDof[(int)*qd];
      shmH1QuadToDofD(qd) = H1QuadToDofD[(int)*qd];
    },


    [=] __device__(ELEM el, NUM_L2DQ dq, NUM_H1QD qd,shmemL2DofToQuad_t shmL2DofToQuad ,shmemH1QuadToDof_t shmH1QuadToDof ,shmemH1QuadToDof_t shmH1QuadToDofD  ,  double &)
  {
    double e_xyz[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      e_xyz[i] = 0;
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      double e_xy[NUM_QUAD_2D];
      for (int i = 0; i < NUM_QUAD_2D; ++i) {
        e_xy[i] = 0;
      }
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        double e_x[NUM_QUAD_1D];
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          e_x[qy] = 0;
        }
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          const double r_e = e[ijklN(dx,dy,dz,(int)*el,L2_DOFS_1D)];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            //e_x[qx] += L2DofToQuad[ijN(qx,dx,NUM_QUAD_1D)] * r_e;
            e_x[qx] += shmL2DofToQuad(convertIndex<NUM_L2DQ,int>(ijN(qx,dx,NUM_QUAD_1D))) * r_e;
          }
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          //const double wy = L2DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
          const double wy = shmL2DofToQuad(convertIndex<NUM_L2DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xy[ijN(qx,qy,NUM_QUAD_1D)] += wy * e_x[qx];
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        //const double wz = L2DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
        const double wz = shmL2DofToQuad(convertIndex<NUM_L2DQ,int>(ijN(qz,dz,NUM_QUAD_1D)));
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)] += wz * e_xy[ijN(qx,qy,NUM_QUAD_1D)];
          }
        }
      }
    }
    for (int c = 0; c < 3; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            v[_ijklmNM(c,dx,dy,dz,(int)*el,NUM_DOFS_1D,numElements)] = 0;
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        double Dxy_x[H1_DOFS_1D * H1_DOFS_1D];
        double xDy_y[H1_DOFS_1D * H1_DOFS_1D];
        double xy_z[H1_DOFS_1D * H1_DOFS_1D] ;
        for (int d = 0; d < (H1_DOFS_1D * H1_DOFS_1D); ++d) {
          Dxy_x[d] = xDy_y[d] = xy_z[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double Dx_x[H1_DOFS_1D];
          double x_y[H1_DOFS_1D];
          double x_z[H1_DOFS_1D];
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            Dx_x[dx] = x_y[dx] = x_z[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_e = e_xyz[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            const double esx = r_e * stressJinvT[ijklmnNM(0,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)];
            const double esy = r_e * stressJinvT[ijklmnNM(1,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)];
            const double esz = r_e * stressJinvT[ijklmnNM(2,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)];
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              //Dx_x[dx] += esx * H1QuadToDofD[ijN(dx,qx,H1_DOFS_1D)];
              //x_y[dx]  += esy * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              //x_z[dx]  += esz * H1QuadToDof[ijN(dx,qx,H1_DOFS_1D)];
              Dx_x[dx] += esx * shmH1QuadToDofD(convertIndex<NUM_H1QD,int>(ijN(dx,qx,H1_DOFS_1D)));
              x_y[dx] += esy * shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dx,qx,H1_DOFS_1D)));
              x_z[dx] += esz * shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dx,qx,H1_DOFS_1D)));
            }
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            //const double wy  = H1QuadToDof[ijN(dy,qy,H1_DOFS_1D)];
            //const double wDy = H1QuadToDofD[ijN(dy,qy,H1_DOFS_1D)];
            const double wy  = shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dy,qy,H1_DOFS_1D)));
            const double wDy = shmH1QuadToDofD(convertIndex<NUM_H1QD,int>(ijN(dy,qy,H1_DOFS_1D)));
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              Dxy_x[ijN(dx,dy,H1_DOFS_1D)] += Dx_x[dx] * wy;
              xDy_y[ijN(dx,dy,H1_DOFS_1D)] += x_y[dx]  * wDy;
              xy_z[ijN(dx,dy,H1_DOFS_1D)]  += x_z[dx]  * wy;
            }
          }
        }
        for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
          //const double wz  = H1QuadToDof[ijN(dz,qz,H1_DOFS_1D)];
          //const double wDz = H1QuadToDofD[ijN(dz,qz,H1_DOFS_1D)];
          const double wz  = shmH1QuadToDof(convertIndex<NUM_H1QD,int>(ijN(dz,qz,H1_DOFS_1D)));
          const double wDz = shmH1QuadToDofD(convertIndex<NUM_H1QD,int>(ijN(dz,qz,H1_DOFS_1D)));
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              v[_ijklmNM(c,dx,dy,dz,(int)*el,NUM_DOFS_1D,numElements)] +=
                ((Dxy_x[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xDy_y[ijN(dx,dy,H1_DOFS_1D)] * wz) +
                 (xy_z[ijN(dx,dy,H1_DOFS_1D)]  * wDz));
            }
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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMultTranspose3D(
#ifndef __TEMPLATES__
                                  const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
#endif
                                  const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
#ifdef __LAMBDA__
  forall(el,numElements,
#else
  const int el = blockDim.x * blockIdx.x + threadIdx.x;
  if (el < numElements)
#endif
  {
    double vStress[NUM_QUAD_3D];
    for (int i = 0; i < NUM_QUAD_3D; ++i) {
      vStress[i] = 0;
    }
    for (int c = 0; c < NUM_DIM; ++c) {
      for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
        double Dxy_x[NUM_QUAD_2D];
        double xDy_y[NUM_QUAD_2D];
        double xy_z[NUM_QUAD_2D] ;
        for (int i = 0; i < NUM_QUAD_2D; ++i) {
          Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
        }
        for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
          double Dx_x[NUM_QUAD_1D];
          double x_y[NUM_QUAD_1D];
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            Dx_x[qx] = x_y[qx] = 0;
          }
          for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
            const double r_v = v[_ijklmNM(c,dx,dy,dz,el,NUM_DOFS_1D,numElements)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
              x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
            }
          }
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dxy_x[ijN(qx,qy,NUM_QUAD_1D)] += Dx_x[qx] * wy;
              xDy_y[ijN(qx,qy,NUM_QUAD_1D)] += x_y[qx]  * wDy;
              xy_z[ijN(qx,qy,NUM_QUAD_1D)]  += x_y[qx]  * wy;
            }
          }
        }
        for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
          const double wz  = H1DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
          const double wDz = H1DofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
          for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] +=
                ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(0,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                 (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(1,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]) +
                 (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz*stressJinvT[ijklmnNM(2,c,qx,qy,qz,el,NUM_DIM,NUM_QUAD_1D)]));
            }
          }
        }
      }
    }
    for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
      for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] = 0;
        }
      }
    }
    for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
      double e_xy[L2_DOFS_1D * L2_DOFS_1D];
      for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d) {
        e_xy[d] = 0;
      }
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
        double e_x[L2_DOFS_1D];
        for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
          e_x[dx] = 0;
        }
        for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
          const double r_v = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
          }
        }
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_xy[ijN(dx,dy,L2_DOFS_1D)] += e_x[dx] * w;
          }
        }
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
        const double w = L2QuadToDof[ijN(dz,qz,L2_DOFS_1D)];
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e[ijklN(dx,dy,dz,el,L2_DOFS_1D)] += w * e_xy[ijN(dx,dy,L2_DOFS_1D)];
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
#ifdef __TEMPLATES__
template<const int NUM_DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D,
         const int L2_DOFS_1D,
         const int H1_DOFS_1D> kernel__
#endif
static void rForceMultTranspose3DNested1(
#ifndef __TEMPLATES__
                                  const int NUM_DIM,
                                  const int NUM_DOFS_1D,
                                  const int NUM_QUAD_1D,
                                  const int L2_DOFS_1D,
                                  const int H1_DOFS_1D,
#endif
                                  const int numElements,
                                  const double* restrict L2QuadToDof,
                                  const double* restrict H1DofToQuad,
                                  const double* restrict H1DofToQuadD,
                                  const double* restrict stressJinvT,
                                  const double* restrict v,
                                  double* restrict e) {
  const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;
  const int NUM_QUAD_3D = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;

  
  auto segments = camp::make_tuple(
    TypedRangeSegment<ELEM>(0,numElements),
    TypedRangeSegment<NUM_L2QD>(0,NUM_QUAD_1D * L2_DOFS_1D),
    TypedRangeSegment<NUM_H1DQ>(0,H1_DOFS_1D * NUM_QUAD_1D)
  );

  //printf("Sizes NUM_QUAD_2D %d  NUM_QUAD_3D %d\n",NUM_QUAD_2D,NUM_QUAD_3D);
  using shmemL2QuadToDof_t = ShmemTile<cuda_shmem, double,ArgList<1>,SizeList<(NUM_QUAD_1D * L2_DOFS_1D)>, decltype(segments)>;
  using shmemH1DofToQuad_t = ShmemTile<cuda_shmem, double,ArgList<2>,SizeList<(H1_DOFS_1D * NUM_QUAD_1D)>, decltype(segments)>;
  shmemL2QuadToDof_t shmL2QuadToDof; 
  shmemH1DofToQuad_t shmH1DofToQuad;
  shmemH1DofToQuad_t shmH1DofToQuadD;

  kernel_param<Pol1>(

    segments, 

    RAJA::make_tuple(
      shmL2QuadToDof,
      shmH1DofToQuad,
      shmH1DofToQuadD,
      0.0), 

    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
    {
      shmL2QuadToDof(qd) = L2QuadToDof[(int)*qd];
    },


    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
    {
      shmH1DofToQuad(dq) = H1DofToQuad[(int)*dq];
      shmH1DofToQuadD(dq) = H1DofToQuadD[(int)*dq];
    },


    [=] __device__(ELEM el, NUM_L2QD qd, NUM_H1DQ dq,shmemL2QuadToDof_t shmL2QuadToDof ,shmemH1DofToQuad_t shmH1DofToQuad ,shmemH1DofToQuad_t shmH1DofToQuadD  ,  double &)
    {
      double vStress[NUM_QUAD_3D];
      for (int i = 0; i < NUM_QUAD_3D; ++i) {
        vStress[i] = 0;
      }
      for (int c = 0; c < NUM_DIM; ++c) {
        for (int dz = 0; dz < H1_DOFS_1D; ++dz) {
          double Dxy_x[NUM_QUAD_2D];
          double xDy_y[NUM_QUAD_2D];
          double xy_z[NUM_QUAD_2D] ;
          for (int i = 0; i < NUM_QUAD_2D; ++i) {
            Dxy_x[i] = xDy_y[i] = xy_z[i] = 0;
          }
          for (int dy = 0; dy < H1_DOFS_1D; ++dy) {
            double Dx_x[NUM_QUAD_1D];
            double x_y[NUM_QUAD_1D];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
              Dx_x[qx] = x_y[qx] = 0;
            }
            for (int dx = 0; dx < H1_DOFS_1D; ++dx) {
              const double r_v = v[_ijklmNM(c,dx,dy,dz,(int)*el,NUM_DOFS_1D,numElements)];
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                //Dx_x[qx] += r_v * H1DofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
                //x_y[qx]  += r_v * H1DofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                x_y[qx] += r_v * shmH1DofToQuad(convertIndex<NUM_H1DQ,int>(ijN(qx,dx,NUM_QUAD_1D)));
                Dx_x[qx]  += r_v * shmH1DofToQuadD(convertIndex<NUM_H1DQ,int>(ijN(qx,dx,NUM_QUAD_1D)));
              }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              //const double wy  = H1DofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
              //const double wDy = H1DofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
              const double wy  = shmH1DofToQuad(convertIndex<NUM_H1DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
              const double wDy = shmH1DofToQuadD(convertIndex<NUM_H1DQ,int>(ijN(qy,dy,NUM_QUAD_1D)));
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                Dxy_x[ijN(qx,qy,NUM_QUAD_1D)] += Dx_x[qx] * wy;
                xDy_y[ijN(qx,qy,NUM_QUAD_1D)] += x_y[qx]  * wDy;
                xy_z[ijN(qx,qy,NUM_QUAD_1D)]  += x_y[qx]  * wy;
              }
            }
          }
          for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
            //const double wz  = H1DofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
            //const double wDz = H1DofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
            const double wz  = shmH1DofToQuad(convertIndex<NUM_H1DQ,int>(ijN(qz,dz,NUM_QUAD_1D)));
            const double wDz = shmH1DofToQuadD(convertIndex<NUM_H1DQ,int>(ijN(qz,dz,NUM_QUAD_1D)));
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
              for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
                vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)] +=
                  ((Dxy_x[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(0,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)]) +
                   (xDy_y[ijN(qx,qy,NUM_QUAD_1D)]*wz *stressJinvT[ijklmnNM(1,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)]) +
                   (xy_z[ijN(qx,qy,NUM_QUAD_1D)] *wDz*stressJinvT[ijklmnNM(2,c,qx,qy,qz,(int)*el,NUM_DIM,NUM_QUAD_1D)]));
              }
            }
          }
        }
      }
      for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
        for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e[ijklN(dx,dy,dz,(int)*el,L2_DOFS_1D)] = 0;
          }
        }
      }
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz) {
        double e_xy[L2_DOFS_1D * L2_DOFS_1D];
        for (int d = 0; d < (L2_DOFS_1D * L2_DOFS_1D); ++d) {
          e_xy[d] = 0;
        }
        for (int qy = 0; qy < NUM_QUAD_1D; ++qy) {
          double e_x[L2_DOFS_1D];
          for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
            e_x[dx] = 0;
          }
          for (int qx = 0; qx < NUM_QUAD_1D; ++qx) {
            const double r_v = vStress[ijkN(qx,qy,qz,NUM_QUAD_1D)];
            for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
              //e_x[dx] += r_v * L2QuadToDof[ijN(dx,qx,L2_DOFS_1D)];
              e_x[dx] += r_v * shmL2QuadToDof(convertIndex<NUM_L2QD ,int>(ijN(dx,qx,L2_DOFS_1D)));
            }
          }
          for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
            //const double w = L2QuadToDof[ijN(dy,qy,L2_DOFS_1D)];
            const double w = shmL2QuadToDof(convertIndex<NUM_L2QD ,int>(ijN(dy,qy,L2_DOFS_1D)));
            for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
              e_xy[ijN(dx,dy,L2_DOFS_1D)] += e_x[dx] * w;
            }
          }
        }
        for (int dz = 0; dz < L2_DOFS_1D; ++dz) {
          //const double w = L2QuadToDof[ijN(dz,qz,L2_DOFS_1D)];
          const double w = shmL2QuadToDof(convertIndex<NUM_L2QD ,int>(ijN(dz,qz,L2_DOFS_1D)));
          for (int dy = 0; dy < L2_DOFS_1D; ++dy) {
            for (int dx = 0; dx < L2_DOFS_1D; ++dx) {
              e[ijklN(dx,dy,dz,(int)*el,L2_DOFS_1D)] += w * e_xy[ijN(dx,dy,L2_DOFS_1D)];
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
typedef void (*fForceMult)(const int numElements,
                           const double* restrict L2QuadToDof,
                           const double* restrict H1DofToQuad,
                           const double* restrict H1DofToQuadD,
                           const double* restrict stressJinvT,
                           const double* restrict e,
                           double* restrict v);

// *****************************************************************************
void rForceMult(const int NUM_DIM,
                const int NUM_DOFS_1D,
                const int NUM_QUAD_1D,
                const int L2_DOFS_1D,
                const int H1_DOFS_1D,
                const int nzones,
                const double* restrict L2QuadToDof,
                const double* restrict H1DofToQuad,
                const double* restrict H1DofToQuadD,
                const double* restrict stressJinvT,
                const double* restrict e,
                double* restrict v) {
  push(Lime);
#ifndef __LAMBDA__
  const int blck = CUDA_BLOCK_SIZE;
  const int grid = (nzones+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  assert(NUM_DOFS_1D==H1_DOFS_1D);
  assert(L2_DOFS_1D==NUM_DOFS_1D-1);
  const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
  assert(LOG2(NUM_DIM)<=4);
  assert(LOG2(NUM_DOFS_1D-2)<=4);
  static std::unordered_map<unsigned long long, fForceMult> call = {
    {0x20,&rForceMult2DNested1<2,2,2,1,2>},
    {0x21,&rForceMult2DNested1<2,3,4,2,3>},
    {0x22,&rForceMult2DNested1<2,4,6,3,4>},
    {0x23,&rForceMult2DNested1<2,5,8,4,5>},
    {0x24,&rForceMult2DNested1<2,6,10,5,6>},
    {0x25,&rForceMult2DNested1<2,7,12,6,7>},
    {0x26,&rForceMult2DNested1<2,8,14,7,8>},
    {0x27,&rForceMult2DNested1<2,9,16,8,9>},
    {0x28,&rForceMult2DNested1<2,10,18,9,10>},
    {0x29,&rForceMult2DNested1<2,11,20,10,11>},
    {0x2A,&rForceMult2DNested1<2,12,22,11,12>},
    {0x2B,&rForceMult2DNested1<2,13,24,12,13>},
    {0x2C,&rForceMult2DNested1<2,14,26,13,14>},
    {0x2D,&rForceMult2DNested1<2,15,28,14,15>},
    {0x2E,&rForceMult2DNested1<2,16,30,15,16>},
    {0x2F,&rForceMult2DNested1<2,17,32,16,17>},
    // 3D
    {0x30,&rForceMult3DNested1<3,2,2,1,2>},
    {0x31,&rForceMult3DNested1<3,3,4,2,3>},
    {0x32,&rForceMult3DNested1<3,4,6,3,4>},
    {0x33,&rForceMult3DNested1<3,5,8,4,5>},
    {0x34,&rForceMult3DNested1<3,6,10,5,6>},
    {0x35,&rForceMult3DNested1<3,7,12,6,7>},
    {0x36,&rForceMult3DNested1<3,8,14,7,8>},
    {0x37,&rForceMult3DNested1<3,9,16,8,9>},
    {0x38,&rForceMult3DNested1<3,10,18,9,10>},
    {0x39,&rForceMult3DNested1<3,11,20,10,11>},
    {0x3A,&rForceMult3DNested1<3,12,22,11,12>},
    {0x3B,&rForceMult3DNested1<3,13,24,12,13>},
    {0x3C,&rForceMult3DNested1<3,14,26,13,14>},
    {0x3D,&rForceMult3DNested1<3,15,28,14,15>},
    {0x3E,&rForceMult3DNested1<3,16,30,15,16>},
    {0x3F,&rForceMult3DNested1<3,17,32,16,17>},
  };
  if (!call[id]){
    printf("\n[rForceMult] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rForceMult,id,grid,blck,
        nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
#else
  if (NUM_DIM==2)
    call0(rForceMult2D,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
  if (NUM_DIM==3)
    call0(rForceMult3D,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,e,v);
  if (NUM_DIM!=2 && NUM_DIM!=3)
    exit(printf("\n[rForceMult] NUM_DIM!=2 && NUM_DIM!=3 ERROR"));    
#endif // __TEMPLATES__
  pop();
}

// *****************************************************************************
typedef void (*fForceMultTranspose)(const int numElements,
                                    const double* restrict L2QuadToDof,
                                    const double* restrict H1DofToQuad,
                                    const double* restrict H1DofToQuadD,
                                    const double* restrict stressJinvT,
                                    const double* restrict v,
                                    double* restrict e);

// *****************************************************************************
void rForceMultTranspose(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int nzones,
                         const double* restrict L2QuadToDof,
                         const double* restrict H1DofToQuad,
                         const double* restrict H1DofToQuadD,
                         const double* restrict stressJinvT,
                         const double* restrict v,
                         double* restrict e) {
  push(Lime);
#ifndef __LAMBDA__
  const int blck = CUDA_BLOCK_SIZE;
  const int grid = (nzones+blck-1)/blck;
#endif
#ifdef __TEMPLATES__
  assert(NUM_DOFS_1D==H1_DOFS_1D);
  assert(L2_DOFS_1D==NUM_DOFS_1D-1);
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  assert(NUM_QUAD_1D==2*(NUM_DOFS_1D-1));
  const unsigned int id = ((NUM_DIM)<<4)|(NUM_DOFS_1D-2);
  static std::unordered_map<unsigned long long, fForceMultTranspose> call = {
    // 2D
    {0x20,&rForceMultTranspose2DNested1<2,2,2,1,2>},
    {0x21,&rForceMultTranspose2DNested1<2,3,4,2,3>},
    {0x22,&rForceMultTranspose2DNested1<2,4,6,3,4>},
    {0x23,&rForceMultTranspose2DNested1<2,5,8,4,5>},
    {0x24,&rForceMultTranspose2DNested1<2,6,10,5,6>},
    {0x25,&rForceMultTranspose2DNested1<2,7,12,6,7>},
    {0x26,&rForceMultTranspose2DNested1<2,8,14,7,8>},
    {0x27,&rForceMultTranspose2DNested1<2,9,16,8,9>},
    {0x28,&rForceMultTranspose2DNested1<2,10,18,9,10>},
    {0x29,&rForceMultTranspose2DNested1<2,11,20,10,11>},
    {0x2A,&rForceMultTranspose2DNested1<2,12,22,11,12>},
    {0x2B,&rForceMultTranspose2DNested1<2,13,24,12,13>},
    {0x2C,&rForceMultTranspose2DNested1<2,14,26,13,14>},
    {0x2D,&rForceMultTranspose2DNested1<2,15,28,14,15>},
    {0x2E,&rForceMultTranspose2DNested1<2,16,30,15,16>},
    {0x2F,&rForceMultTranspose2DNested1<2,17,32,16,17>},
    // 3D
    {0x30,&rForceMultTranspose3DNested1<3,2,2,1,2>},
    {0x31,&rForceMultTranspose3DNested1<3,3,4,2,3>},
    {0x32,&rForceMultTranspose3DNested1<3,4,6,3,4>},
    {0x33,&rForceMultTranspose3DNested1<3,5,8,4,5>},
    {0x34,&rForceMultTranspose3DNested1<3,6,10,5,6>},
    {0x35,&rForceMultTranspose3DNested1<3,7,12,6,7>},
    {0x36,&rForceMultTranspose3DNested1<3,8,14,7,8>},
    {0x37,&rForceMultTranspose3DNested1<3,9,16,8,9>},
    {0x38,&rForceMultTranspose3DNested1<3,10,18,9,10>},
    {0x39,&rForceMultTranspose3DNested1<3,11,20,10,11>},
    {0x3A,&rForceMultTranspose3DNested1<3,12,22,11,12>},
    {0x3B,&rForceMultTranspose3DNested1<3,13,24,12,13>},
    {0x3C,&rForceMultTranspose3DNested1<3,14,26,13,14>},
    {0x3D,&rForceMultTranspose3DNested1<3,15,28,14,15>},
    {0x3E,&rForceMultTranspose3DNested1<3,16,30,15,16>},
    {0x3F,&rForceMultTranspose3DNested1<3,17,32,16,17>},
  };
  if (!call[id]) {
    printf("\n[rForceMultTranspose] id \033[33m0x%X\033[m ",id);
    fflush(stdout);
  }
  assert(call[id]);
  call0(rForceMultTranspose,id,grid,blck,
        nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
#else
  if (NUM_DIM==2)
    call0(rForceMultTranspose2D,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
  if (NUM_DIM==3)
    call0(rForceMultTranspose3D,id,grid,blck,
          NUM_DIM,NUM_DOFS_1D,NUM_QUAD_1D,L2_DOFS_1D,H1_DOFS_1D,
          nzones,L2QuadToDof,H1DofToQuad,H1DofToQuadD,stressJinvT,v,e);
  if (NUM_DIM!=2 && NUM_DIM!=3)
    exit(printf("\n[rForceMultTranspose] NUM_DIM!=2 && NUM_DIM!=3 ERROR"));  
#endif
  pop(); 
}

