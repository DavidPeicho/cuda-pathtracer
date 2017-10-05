#pragma once

#include <cuda.h>
#include <driver_types.h>
#include <string>

#define cudaCheckError()                                \
{                                                       \
  cudaError_t e = cudaGetLastError();                   \
  if(e != cudaSuccess)                                  \
  {                                                     \
      printf("Cuda failure %s:%d: '%s'\n",              \
             __FILE__,__LINE__,cudaGetErrorString(e));  \
      exit(EXIT_FAILURE);                               \
  }                                                     \
}

void checkDevice(struct cudaDeviceProp* device);
