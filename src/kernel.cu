#include <cuda_runtime.h>
#include <stdbool.h>

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

__global__
void
kernel(const int width, const int height)
{
  const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int x   = idx % width;
  const int y   = idx / width;

  union rgba_24 rgbx;

  rgbx.r = 0;
  rgbx.g = idx % 125;
  rgbx.b = idx % 255;
  rgbx.a = 255;

  surf2Dwrite(rgbx.b32,
    surf,
    x*sizeof(rgbx),
    y,
    cudaBoundaryModeZero);
}

cudaError_t
kernel_launcher(cudaArray_const_t array, const int width, const int height, cudaEvent_t event, cudaStream_t stream)
{
  cudaBindSurfaceToArray(surf, array);

  const unsigned int threads_per_block = 256;
  const unsigned int blocks = (width * height + threads_per_block - 1) / threads_per_block;

  if (blocks > 0)
    kernel<<<blocks, threads_per_block, 0, stream>>>(width, height);
  
  return cudaSuccess;
}