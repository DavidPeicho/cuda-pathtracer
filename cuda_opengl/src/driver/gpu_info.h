#pragma once

#include <string>
#include <sstream>

namespace driver
{

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
      std::string
      getProfile();

    public:
      inline GPU&
      getGLGPU()
      {
        return *_gpus[0];
      }

      inline GPU&
      getCUDAGPU()
      {
        return *_gpus[1];
      }

    private:
      static GPUInfo* _instance;

    private:
      GPU *_gpus[2];
  };

} // namespace driver
