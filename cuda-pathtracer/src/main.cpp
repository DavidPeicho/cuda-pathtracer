#include <cuda.h>
#include <iostream>

#include "cuda_helper.h"
#include "gpu_info.h"

int main()
{
  struct cudaDeviceProp deviceProp;
  checkDevice(&deviceProp);
  GPUInfo::instance()->setNbThreads(deviceProp.maxThreadsPerBlock);

  std::cout << GPUInfo::instance()->getNbThreads() << std::endl;

  std::cin.get();

  return 0;
}