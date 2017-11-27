#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <string>

namespace driver {
/// <summary>
/// Encapsulates information about an used GPU.
/// </summary>
class GPUInfo
{
  struct GPU
  {
    int device_id;
    int clock_rate;
    int warp_size;
    int regs_per_block;
    int shared_mem_block;
    int multiproc_count;
    int max_threads_per_block;
    int max_threads_dim[3];
  };

public:
  GPUInfo();
  ~GPUInfo();

public:
  std::string getProfile();

public:
  inline GPU& getGLGPU() { return *_gpus[0]; }

  inline GPU& getCUDAGPU() { return *_gpus[1]; }

  inline size_t getFreeMo()
  {
    size_t free_byte = 0;
    size_t total_byte = 0;

    cudaMemGetInfo(&free_byte, &total_byte);
    return free_byte / (1024 * 1024);
  }

private:
  static GPUInfo* _instance;

private:
  GPU* _gpus[2];
};

} // namespace driver
