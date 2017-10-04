#include <stdio.h>
#include <iostream>

#define cudaCheckError() {                                   \
        cudaError_t e=cudaGetLastError();                    \
        if(e!=cudaSuccess) {                                 \
            printf("Cuda failure %s:%d: '%s'\n",             \
                   __FILE__,__LINE__,cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    }


void checkDevice()
{
  //cudaCheckError();
	int nDevices;
  cudaGetDeviceCount(&nDevices);
  std::cout << nDevices << std::endl;
  //œstd::cout << nDevices << std::endl;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

