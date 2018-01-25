blt_add_library(
  NAME laghos_raja
  SOURCES rbilinearform.cpp
  rbilininteg.cpp
  rfespace.cpp
  rgridfunc.cpp
  rmalloc.cpp
  rvector.cpp
  DEPENDS_ON laghos_raja_kernels ${LAGHOS_DEPENDS}
  )
