#pragma once

#ifdef _WIN64
# include <windows.h>
#endif

#include <cuda_runtime.h>
#include <glad/glad.h>

#include <cuda_gl_interop.h>

namespace driver
{
  class Interop
  {
    public:
      Interop(unsigned int w, unsigned int h);
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
      setSize(const unsigned int w, const unsigned int h);

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
      getSize(unsigned int& w, unsigned int& h);

      inline unsigned width () const { return _width; }
      inline unsigned half_width () const { return _half_width; }
      inline unsigned height() const { return _height; }
      inline unsigned half_height() const { return _half_height; }

    private:
      unsigned int _width;
      unsigned int _half_width;
      unsigned int _height;
      unsigned int _half_height;

      bool _allocated;

      int _index;
      GLuint _fb[2];
      GLuint _rb[2];
      cudaGraphicsResource *_d_cgr[2];
      cudaArray *_d_ca[2];
  };
} // namespace driver
