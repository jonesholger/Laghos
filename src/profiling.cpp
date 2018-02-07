#include "profiling.hpp"
#ifdef LAGHOS_ENABLE_CALIPER
#include <caliper/cali.h>
#endif
void enableCudaProfiling(){
#ifdef LAGHOS_ENABLE_CALIPER
  cali_config_preset("CALI_SERVICES_ENABLE","event:trace:nvprof:mpi");
#else
  std::cerr << "Warning, attempted to enable CUDA profiling without Caliper, ranges will not be visible\n";
#endif
}

