#ifdef _WIN64
# include <windows.h>
#endif

#include <cstring>
#include <cuda_gl_interop.h>
#include <stdexcept>

#include <driver/gpu_info.h>

namespace driver
{
  GPUInfo* GPUInfo::_instance = nullptr;

  GPUInfo::GPUInfo()
  {
    int devices_count;
    cudaGetDeviceCount(&devices_count);
    if (devices_count <= 0)
    {
      std::string error = "GPUInfo(): no GPU found, or connection failed.";
      throw std::runtime_error(error);
    }

    // Gets back the GPU properties.
    struct GPU *gpu = new GPU;
    struct cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    gpu->device_id = 0;
    gpu->clock_rate = deviceProp.clockRate;
    gpu->warp_size = deviceProp.warpSize;
    gpu->regs_per_block = deviceProp.regsPerBlock;
    gpu->shared_mem_block = deviceProp.sharedMemPerBlock;
    gpu->max_threads_per_block = deviceProp.maxThreadsPerBlock;
    gpu->multiproc_count = deviceProp.multiProcessorCount;
    std::memcpy(&gpu->max_threads_dim, &deviceProp.maxThreadsDim, 3 * sizeof (int));

    // For now, we only handle one GPU It would be nice to handle two or more
    // GPUs, and to choose them according to some heuristics.
    // For instance, it would be nice to render the quad on the slow GPU,
    // and to pathtrace with CUDA on the fast GPU.
    _gpus[0] = gpu;
    _gpus[1] = gpu;
  }

  GPUInfo::~GPUInfo()
  {
    GPU *adress = _gpus[0];

    delete _gpus[0];
    // Both GPUs are pointing to the same device.
    if (adress == _gpus[1]) return;

    delete _gpus[1];
  }

  std::string
  GPUInfo::getProfile()
  {
    std::stringstream ss;
    ss << "[GPU " << _gpus[0]->device_id << "]";

    size_t free_byte;
    size_t total_byte;

    if (cudaMemGetInfo(&free_byte, &total_byte) == cudaSuccess)
    {
      free_byte /= 1024;
      total_byte /= 1024;
      ss << "\n- memory: " << free_byte << "/" << total_byte << " (ko)";
    }
    ss << std::endl;

    return ss.str();
  }

} // namespace driver
