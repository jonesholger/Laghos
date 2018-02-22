
include_directories(.)
include_directories(./raja)
include_directories(./raja/config)
include_directories(./raja/fem)
include_directories(./raja/general)
include_directories(./raja/kernels)
include_directories(./raja/kernels/include)
include_directories(./raja/linalg)
include_directories(./raja/tests)


add_subdirectory(raja)
add_subdirectory(raja/kernels)

set(LAGHOS_EXECUTABLE_DEPENDS
    laghos_raja
    laghos_raja_kernels)

list(APPEND LAGHOS_EXECUTABLE_DEPENDS ${LAGHOS_DEPENDS}) 

blt_add_executable(
  NAME laghos.exe
  SOURCES laghos.cpp 
  laghos_assembly.cpp
  laghos_solver.cpp
  DEPENDS_ON ${LAGHOS_EXECUTABLE_DEPENDS}
  )


