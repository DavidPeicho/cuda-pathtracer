#ifdef _WIN64
# include <windows.h>
#endif

#include <cuda_gl_interop.h>

#include "gpu_info.h"

namespace driver
{

  GPUInfo* GPUInfo::_instance = nullptr;

  GPUInfo::GPUInfo()
  {
    unsigned int devices_count;
    struct GPU *gpu = new GPU;
    cudaGLGetDevices(&devices_count, &gpu->device_id, 1, cudaGLDeviceListAll);

    // Gets back the GPU properties.
    struct cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(deviceProp, 0);

    // For now, we only handle one GPU It would be nice to handle two or more
    // GPUs, and to choose them according to some heuristics.
    // For instance, it would be nice to render the quad on the slow GPU,
    // and to pathtrace with CUDA on the fast GPU.
    _gpus[0] = gpu;
    _gpus[1] = gpu;
  }

} // namespace driver
