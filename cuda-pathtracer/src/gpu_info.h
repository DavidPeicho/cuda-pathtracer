#pragma once

class GPUInfo
{
  public:
    static inline GPUInfo*
    instance()
    {
      if (!_instance)
        _instance = new GPUInfo();
      return _instance;
    }

  public:
    GPUInfo();
    
  public:
    inline void
    setGPUId(int id) { _gpu_id = id; }
    
    inline void
    setNbThreads(int nb_threads) { _nb_threads = nb_threads; }

    inline unsigned int
    getNbThreads() { return _nb_threads; }

  private:
    static GPUInfo* _instance;

  private:
    int _gpu_id;
    int _nb_threads;
};
