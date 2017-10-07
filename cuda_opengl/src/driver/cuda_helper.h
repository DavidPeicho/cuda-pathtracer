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

#define cudaThrowError()                                                      \
{                                                                             \
  cudaError_t e = cudaGetLastError();                                         \
  if(e != cudaSuccess)                                                        \
  {                                                                           \
      std::stringstream ss;                                                   \
      ss << "Cuda failure " << __FILE__  << ":" << __LINE__;                  \
      ss << " : " << cudaGetErrorString(e) << "\n";                           \
      throw std::runtime_error(ss.str());                                     \
  }                                                                           \
}

#define cudaCalloc(DST, NEMB, SIZE)                                         \
{                                                                           \
  {                                                                         \
    cudaError_t __cudaCalloc_err = cudaMalloc(DST, NEMB * SIZE);            \
    if (__cudaCalloc_err == cudaSuccess) cudaMemset(*DST, 0, NEMB * SIZE);  \
  }                                                                         \
}

void checkDevice(struct cudaDeviceProp* device);
