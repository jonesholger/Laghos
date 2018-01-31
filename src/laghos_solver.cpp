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

#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   LAGHOS_MARK_FUNCTION
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAcl";
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

// ***************************************************************************
// * LagrangianHydroOperator
// ***************************************************************************
LagrangianHydroOperator::LagrangianHydroOperator(int size,
                                                 RajaFiniteElementSpace &h1_fes,
                                                 RajaFiniteElementSpace &l2_fes,
                                                 Array<int> &essential_tdofs,
                                                 RajaGridFunction &rho0,
                                                 int source_type_, double cfl_,
                                                 Coefficient *material_,
                                                 bool visc, bool pa,
                                                 double cgt, int cgiter)
   : RajaTimeDependentOperator(size),
     H1FESpace(h1_fes), L2FESpace(l2_fes),
     H1compFESpace(h1_fes.GetParMesh(), h1_fes.FEColl(), 1),
     ess_tdofs(essential_tdofs),
     dim(h1_fes.GetMesh()->Dimension()),
     nzones(h1_fes.GetMesh()->GetNE()),
     l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
     h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
     source_type(source_type_), cfl(cfl_),
     use_viscosity(visc), p_assembly(pa), cg_rel_tol(cgt), cg_max_iter(cgiter),
     material_pcf(material_),
     integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(),
                             3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
     quad_data(dim, nzones, integ_rule.GetNPoints()),
     quad_data_is_current(false),
     VMassPA(H1compFESpace, integ_rule, &quad_data),
     EMassPA(L2FESpace, integ_rule, &quad_data),
     ForcePA(H1FESpace, L2FESpace, integ_rule, &quad_data),
     locCG(), timer()
{
   Vector rho0_ = rho0;
   GridFunction rho0_gf(&L2FESpace, rho0_.GetData());
   GridFunctionCoefficient rho_coeff(&rho0_gf);

   // Initial local mesh size (assumes similar cells).
   double loc_area = 0.0, glob_area;
   int loc_z_cnt = nzones, glob_z_cnt;
   ParMesh *pm = H1FESpace.GetParMesh();
   for (int i = 0; i < nzones; i++) { loc_area += pm->GetElementVolume(i); }
   MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
   MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
   switch (pm->GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT:
         quad_data.h0 = glob_area / glob_z_cnt; break;
      case Geometry::SQUARE:
         quad_data.h0 = sqrt(glob_area / glob_z_cnt); break;
      case Geometry::TRIANGLE:
         quad_data.h0 = sqrt(2.0 * glob_area / glob_z_cnt); break;
      case Geometry::CUBE:
         quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0); break;
      case Geometry::TETRAHEDRON:
         quad_data.h0 = pow(6.0 * glob_area / glob_z_cnt, 1.0/3.0); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   quad_data.h0 /= (double) H1FESpace.GetOrder(0);

   quad_data.dqMaps = RajaDofQuadMaps::Get(H1FESpace,integ_rule);
   quad_data.geom = RajaGeometry::Get(H1FESpace,integ_rule);
   quad_data.Jac0inv = quad_data.geom->invJ;

   RajaVector rhoValues;
   rho0.ToQuad(integ_rule, rhoValues);

   if (dim==1) { assert(false); }
   const int NUM_QUAD = integ_rule.GetNPoints();

   rInitQuadratureData(NUM_QUAD,
                       nzones,
                       (const double*)rhoValues.ptr(),
                       (const double*)quad_data.geom->detJ.ptr(),
                       (const double*)quad_data.dqMaps->quadWeights.ptr(),
                       (double*)quad_data.rho0DetJ0w.ptr());

   // Needs quad_data.rho0DetJ0w
   ForcePA.Setup();
   VMassPA.Setup();
   EMassPA.Setup();

   locCG.SetOperator(EMassPA);
   locCG.iterative_mode = false;
   locCG.SetRelTol(1e-8);
   locCG.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
   locCG.SetMaxIter(200);
   locCG.SetPrintLevel(0);
}

// *****************************************************************************
LagrangianHydroOperator::~LagrangianHydroOperator() {}

// *****************************************************************************
void LagrangianHydroOperator::Mult(const RajaVector &S, RajaVector &dS_dt) const
{
   LAGHOS_MARK_FUNCTION
   dS_dt = 0.0;

   // Make sure that the mesh positions correspond to the ones in S. This is
   // needed only because some mfem time integrators don't update the solution
   // vector at every intermediate stage (hence they don't change the mesh).
   Vector h_x = RajaVector(S.GetRange(0, H1FESpace.GetVSize()));
   ParGridFunction x(&H1FESpace, h_x.GetData());
   H1FESpace.GetParMesh()->NewNodes(x, false);
   
   UpdateQuadratureData(S);

   // The monolithic BlockVector stores the unknown fields as follows:
   // - Position
   // - Velocity
   // - Specific Internal Energy

   const int VsizeL2 = L2FESpace.GetVSize();
   const int VsizeH1 = H1FESpace.GetVSize();

   RajaVector v = S.GetRange(VsizeH1, VsizeH1);
   RajaVector e = S.GetRange(2*VsizeH1, VsizeL2);
   
   RajaVector dx = dS_dt.GetRange(0, VsizeH1);
   RajaVector dv = dS_dt.GetRange(VsizeH1, VsizeH1);
   RajaVector de = dS_dt.GetRange(2*VsizeH1, VsizeL2);

   // Set dx_dt = v (explicit)
   dx = v;
   
   // Solve for velocity.
   RajaVector one(VsizeL2), rhs(VsizeH1); one = 1.0;
   timer.sw_force.Start();
   ForcePA.Mult(one, rhs);
   timer.sw_force.Stop();
   timer.dof_tstep += H1FESpace.GlobalTrueVSize();
   rhs.Neg();

   //RajaVector B(H1compFESpace.GetTrueVSize()); B=0.0;
   //RajaVector X(H1compFESpace.GetTrueVSize()); X=0.0;
   //dv = 0.0;

   // Partial assembly solve for each velocity component.
   const int size = H1compFESpace.GetVSize();
   for (int c = 0; c < dim; c++)
   {
      RajaVector rhs_c = rhs.GetRange(c*size, size),
        dv_c  = dv.GetRange(c*size, size);

      Array<int> c_tdofs;
      Array<int> ess_bdr(H1FESpace.GetMesh()->bdr_attributes.Max());
      // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
      // we must enforce v_x/y/z = 0 for the velocity components.
      ess_bdr = 0; ess_bdr[c] = 1;
      // Essential true dofs as if there's only one component.
      H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs);
      
      dv_c = 0.0;
      RajaVector B(H1compFESpace.GetTrueVSize()), X(H1compFESpace.GetTrueVSize());
      H1compFESpace.GetProlongationOperator()->MultTranspose(rhs_c, B);
      H1compFESpace.GetRestrictionOperator()->Mult(dv_c, X);

      VMassPA.SetEssentialTrueDofs(c_tdofs);
      VMassPA.EliminateRHS(B);
      
      TCGSolver<RajaVector> cg(H1FESpace.GetParMesh()->GetComm());
      cg.SetOperator(VMassPA);
      cg.SetRelTol(cg_rel_tol);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(cg_max_iter);
      cg.SetPrintLevel(-1);
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1dof_iter += cg.GetNumIterations() *
        H1compFESpace.GlobalTrueVSize();
      H1compFESpace.GetProlongationOperator()->Mult(X, dv_c);
   }

   // Solve for energy, assemble the energy source if such exists.
   LinearForm *e_source = NULL;
   if (source_type == 1) // 2D Taylor-Green.
   {
      e_source = new LinearForm(&L2FESpace);
      TaylorCoefficient coeff;
      DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
      e_source->AddDomainIntegrator(d);
      e_source->Assemble();
   }
   Array<int> l2dofs;
   RajaVector e_rhs(VsizeL2), loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
   timer.sw_force.Start();
   ForcePA.MultTranspose(v, e_rhs);
   timer.sw_force.Stop();
   timer.dof_tstep += L2FESpace.GlobalTrueVSize();

   if (e_source) e_rhs += RajaVector(*e_source);
   
   {
     TCGSolver<RajaVector> cg(L2FESpace.GetParMesh()->GetComm());
     cg.SetOperator(EMassPA);
     cg.iterative_mode = false;
     cg.SetRelTol(1e-8);
     cg.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
     cg.SetMaxIter(200);
     cg.SetPrintLevel(-1);
     timer.sw_cgL2.Start();
     cg.Mult(e_rhs, de);
     timer.sw_cgL2.Stop();
     timer.L2dof_iter += cg.GetNumIterations() * L2FESpace.TrueVSize();
   }
   delete e_source;

   quad_data_is_current = false;
}

double LagrangianHydroOperator::GetTimeStepEstimate(const RajaVector &S) const
{
   LAGHOS_MARK_FUNCTION
   Vector h_x = RajaVector(S.GetRange(0, H1FESpace.GetVSize()));
   ParGridFunction x(&H1FESpace, h_x.GetData());
   H1FESpace.GetMesh()->NewNodes(x, false);
   UpdateQuadratureData(S);

   double glob_dt_est;
   MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                 H1FESpace.GetParMesh()->GetComm());
   return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
   quad_data.dt_est = numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho)
{
   LAGHOS_MARK_FUNCTION
   rho.SetSpace(&L2FESpace);

   DenseMatrix Mrho(l2dofs_cnt);
   Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   MassIntegrator mi(&integ_rule);
   DensityIntegrator di(quad_data,integ_rule);
   for (int i = 0; i < nzones; i++)
   {
      di.AssembleRHSElementVect(*L2FESpace.GetFE(i),
                                *L2FESpace.GetElementTransformation(i), rhs);
      mi.AssembleElementMatrix(*L2FESpace.GetFE(i),
                               *L2FESpace.GetElementTransformation(i), Mrho);
      inv.Factor();
      inv.Mult(rhs, rho_z);
      L2FESpace.GetElementDofs(i, dofs);
      rho.SetSubVector(dofs, rho_z);
   }
}

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps)
{
   double my_rt[5], rt_max[5];
   my_rt[0] = timer.sw_cgH1.RealTime();
   my_rt[1] = timer.sw_cgL2.RealTime();
   my_rt[2] = timer.sw_force.RealTime();
   my_rt[3] = timer.sw_qdata.RealTime();
   my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
   MPI_Reduce(my_rt, rt_max, 5, MPI_DOUBLE, MPI_MAX, 0, H1FESpace.GetComm());

   double mydata[2], alldata[2];
   mydata[0] = timer.L2dof_iter;
   mydata[1] = timer.quad_tstep;
   MPI_Reduce(mydata, alldata, 2, MPI_DOUBLE, MPI_SUM, 0, H1FESpace.GetComm());

   if (IamRoot)
   {
      using namespace std;
      cout << endl;
      cout << "CG (H1) total time: " << rt_max[0] << endl;
      cout << "CG (H1) rate (megadofs x cg_iterations / second): "
           << 1e-6 * timer.H1dof_iter / rt_max[0] << endl;
      cout << endl;
      cout << "CG (L2) total time: " << rt_max[1] << endl;
      cout << "CG (L2) rate (megadofs x cg_iterations / second): "
           << 1e-6 * alldata[0] / rt_max[1] << endl;
      cout << endl;
      cout << "Forces total time: " << rt_max[2] << endl;
      cout << "Forces rate (megadofs x timesteps / second): "
           << 1e-6 * timer.dof_tstep / rt_max[2] << endl;
      cout << endl;
      cout << "UpdateQuadData total time: " << rt_max[3] << endl;
      cout << "UpdateQuadData rate (megaquads x timesteps / second): "
           << 1e-6 * alldata[1] / rt_max[3] << endl;
      cout << endl;
      cout << "Major kernels total time (seconds): " << rt_max[4] << endl;
      cout << "Major kernels total rate (megadofs x time steps / second): "
           << 1e-6 * H1FESpace.GlobalTrueVSize() * steps / rt_max[4] << endl;
   }
}

// *****************************************************************************
void LagrangianHydroOperator::UpdateQuadratureData(const RajaVector &S) const
{
   if (quad_data_is_current) { return; }
   timer.sw_qdata.Start();
   const int nqp = integ_rule.GetNPoints();

   const int vSize = H1FESpace.GetVSize();
   const int eSize = L2FESpace.GetVSize();
   RajaGridFunction v(H1FESpace, S.GetRange(vSize, vSize));
   RajaGridFunction e(L2FESpace, S.GetRange(2*vSize, eSize));
   quad_data.geom = RajaGeometry::Get(H1FESpace,integ_rule);
   RajaVector v2(H1FESpace.GetVDim() * H1FESpace.GetLocalDofs() * nzones);
   H1FESpace.GlobalToLocal(v, v2);
   RajaVector eValues;
   e.ToQuad(integ_rule, eValues);
   
   const int NUM_QUAD = integ_rule.GetNPoints();
   const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, integ_rule.GetOrder());
   const int NUM_QUAD_1D  = ir1D.GetNPoints();
   const int NUM_DOFS_1D  = H1FESpace.GetFE(0)->GetOrder()+1;

   ElementTransformation *T = H1FESpace.GetElementTransformation(0);
   const IntegrationPoint &ip = integ_rule.IntPoint(0);
   const double gamma = material_pcf->Eval(*T, ip);
   
   rUpdateQuadratureData(gamma,
                         quad_data.h0,
                         cfl,
                         use_viscosity,
                         dim,
                         NUM_QUAD,
                         NUM_QUAD_1D,
                         NUM_DOFS_1D,
                         nzones,
                         quad_data.dqMaps->dofToQuad,
                         quad_data.dqMaps->dofToQuadD,
                         quad_data.dqMaps->quadWeights,
                         v2,
                         eValues,
                         quad_data.rho0DetJ0w,
                         quad_data.Jac0inv,
                         quad_data.geom->J,
                         quad_data.geom->invJ,
                         quad_data.geom->detJ,
                         quad_data.stressJinvT,
                         quad_data.dtEst);
   
   quad_data.dt_est = quad_data.dtEst.Min();
   
   quad_data_is_current = true;
   
   timer.sw_qdata.Stop();
   timer.quad_tstep += nzones * nqp;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
