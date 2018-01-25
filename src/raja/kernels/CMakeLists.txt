blt_add_library(
  NAME laghos_raja_kernels
  SOURCES rForce.cpp
  rGlobalToLocal.cpp
  rGridFuncToQuad.cpp
  rInitGeom.cpp
  rLocalToGlobal.cpp
  rMapping.cpp
  rMassAssemble.cpp
  rMassMultAdd.cpp
  rQuadratureData.cpp
  vector_axpy.cpp
  vector_clear_dofs.cpp
  vector_dot.cpp
  vector_get_subvector.cpp
  vector_map_dofs.cpp
  vector_min.cpp
  vector_neg.cpp
  vector_op_eq.cpp
  vector_set_subvector.cpp
  vector_set_subvector_const.cpp
  vector_vec_add.cpp
  vector_vec_mul.cpp
  vector_vec_sub.cpp
  vector_xpay.cpp
  vector_xsy.cpp
  DEPENDS_ON ${LAGHOS_DEPENDS}
  )
