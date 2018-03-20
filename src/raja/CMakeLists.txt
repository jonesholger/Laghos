blt_add_library(
  NAME laghos_raja
  SOURCES config/rconfig.cpp
  config/rnvvp.cpp
  fem/rbilinearform.cpp
  fem/rbilininteg.cpp
  fem/rconform.cpp
  fem/rfespace.cpp
  fem/rgridfunc.cpp
  fem/rprolong.cpp
  fem/rrestrict.cpp
  general/rcommd.cpp
  general/rmemcpy.cpp
  general/rtable.cpp
  linalg/rsolvers.cpp
  linalg/rvector.cpp
  tests/dot.cpp
  tests/lambda.cpp
  tests/mult.cpp 
  DEPENDS_ON laghos_raja_kernels ${LAGHOS_DEPENDS}
  )
