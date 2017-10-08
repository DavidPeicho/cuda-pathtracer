#include <cuda_runtime.h>
#include <stdbool.h>

#include "utils.h"

surface<void,cudaSurfaceType2D> surf;

union rgba_24
{
  uint1 b32;

  struct
  {
    unsigned  r  : 8;
    unsigned  g  : 8;
    unsigned  b  : 8;
    unsigned  a : 8;
  };
};

__global__ void
kernel(const int width, const int height)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  union rgba_24 rgbx;

  rgbx.r = 0;
  rgbx.g = x % 125;
  rgbx.b = y % 255;
  rgbx.a = 255;

  surf2Dwrite(rgbx.b32,
    surf,
    x*sizeof(rgbx),
    y,
    cudaBoundaryModeZero);
}

cudaError_t
raytrace(cudaArray_const_t array, const unsigned int width,
  const unsigned int height, cudaStream_t stream)
{
  cudaBindSurfaceToArray(surf, array);

  // Register occupancy : nb_threads = regs_per_block / 32
  // Shared memory occupancy : nb_threads = shared_mem / 32
  // Block size occupancy 

  // TODO: We should get into account GPU info, such as number of registers,
  // shared memory size, warp size, etc...
  dim3 threads_per_block(8, 8);
  dim3 nb_blocks(width / 8, height / 8);

  if (nb_blocks.x > 0 && nb_blocks.y > 0)
    kernel<<<nb_blocks, threads_per_block, 0, stream>>>(width, height);
  
  return cudaSuccess;
}