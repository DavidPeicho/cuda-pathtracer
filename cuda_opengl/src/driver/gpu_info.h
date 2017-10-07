#pragma once

namespace driver
{

  class GPUInfo
  {
    struct GPU
    {
      int device_id;
      int clock_rate;
      int max_threads_per_block;
    };

    public:
      GPUInfo();

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
