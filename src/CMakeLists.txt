
include_directories(.)
include_directories(./raja)
include_directories(./raja/kernels)

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
  profiling.cpp
  DEPENDS_ON ${LAGHOS_EXECUTABLE_DEPENDS}
  )


