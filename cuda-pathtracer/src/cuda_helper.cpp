#include "cuda_helper.h"

void checkDevice(struct cudaDeviceProp* deviceProp)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  cudaCheckError();

  if (nDevices <= 0)
  {
    std::string error = "your GPU does not support CUDA ";
    error += "or the drivers are out of date.";
    throw std::runtime_error("checkDevice(): " + error);
  }

  cudaGetDeviceProperties(deviceProp, 0);
}
