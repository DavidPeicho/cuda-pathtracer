#pragma once

#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace driver
{
  /// <summary>
  /// Encapsulates the logic to communicate between
  /// CUDA and OpenGL
  /// </summary>
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

      /// <summary>
      /// Swaps the framebuffers for double buffering.
      /// </summary>
      void
      swap();

      void
      clear();

      /// <summary>
      /// Copies the data of the front framebuffer
      /// to the framebuffer 0 (the screen)
      /// </summary>
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

      /// <summary>
      /// Framebuffers for double buffering.
      /// </summary>
      GLuint _fb[2];
      /// <summary>
      /// RenderBuffers to write in frambuffers.
      /// </summary>
      GLuint _rb[2];

      /// <summary>
      /// CUDA resources making the link between the CUDA
      /// memory and the OpenGL memory.
      /// </summary>
      cudaGraphicsResource *_d_cgr[2];
      cudaArray *_d_ca[2];
  };
} // namespace driver
