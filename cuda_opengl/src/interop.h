#pragma once

#include <cuda_runtime.h>

namespace driver
{
  class Interop
  {
    public:
      Interop(int w, int h);
      Interop();
      ~Interop();

    public:
      cudaError_t
      map(cudaStream_t stream);

      cudaError_t
      unmap(cudaStream_t stream);

      cudaError_t
      clean();

      void
      swap();

      void
      clear();

      void
      blit();

      cudaError_t
      setSize(const int w, const int h);

      inline int
      getIndex()
      {
        return _index;
      }

      inline cudaArray_const_t
      getArray()
      {
        return _d_ca[_index];
      }

      void
      getSize(int& const w, int& const h);

    private:
      int _width;
      int _height;

      bool _allocated;
      
      int _index;
      GLuint _fb[2];
      GLuint _rb[2];
      cudaGraphicsResource *_d_cgr[2];
      cudaArray *_d_ca[2];
  };
} // namespace driver
