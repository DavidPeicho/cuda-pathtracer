#include <iostream>

#include "interop.h"

namespace driver
{
  Interop::Interop(unsigned int w, unsigned int h)
          : _width(w)
          , _height(h)
          , _allocated(false)
          , _index(0)
  {
    glCreateRenderbuffers(2, &_rb[0]);
    glCreateFramebuffers(2, &_fb[0]);

    glNamedFramebufferRenderbuffer(_fb[0], GL_COLOR_ATTACHMENT0,
        GL_RENDERBUFFER, _rb[0]);
    glNamedFramebufferRenderbuffer(_fb[1], GL_COLOR_ATTACHMENT0,
      GL_RENDERBUFFER, _rb[1]);

    _d_cgr[0] = nullptr;
    _d_cgr[1] = nullptr;
    _d_ca[0] = nullptr;
    _d_ca[1] = nullptr;

    this->setSize(w, h);

    _allocated = true;
  }

  Interop::Interop()
    : Interop(0, 0)
  { }

  Interop::~Interop()
  {
    clean();
  }

  cudaError_t
  Interop::map(cudaStream_t stream)
  {
    return cudaGraphicsMapResources(1, &_d_cgr[_index], stream);
  }

  cudaError_t
  Interop::unmap(cudaStream_t stream)
  {
    return cudaGraphicsUnmapResources(1, &_d_cgr[_index], stream);
  }

  void
  Interop::swap()
  {
    _index = (_index + 1) % 2;
  }

  void
  Interop::clear()
  {
    const GLfloat clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glClearNamedFramebufferfv(_fb[_index], GL_COLOR, 0, clear_color);
  }

  void
  Interop::blit()
  {
    glBlitNamedFramebuffer(_fb[_index], 0, 0, 0, _width, _height, 0,
      _height, _width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  }

  cudaError_t
  Interop::clean()
  {
    cudaError_t cuda_err = cudaSuccess;

    if (!_allocated)
      return cuda_err;

    for (int i = 0; i < 2; i++)
    {
      if (_d_cgr[i] != NULL)
        cuda_err = cudaGraphicsUnregisterResource(_d_cgr[i]);
    }

    glDeleteRenderbuffers(2, _rb);
    glDeleteFramebuffers(2, _fb);

    return cuda_err;
  }

  cudaError_t
  Interop::setSize(const unsigned int w, const unsigned int h)
  {
    cudaError_t cuda_err = cudaSuccess;

    _width = w;
    _height = h;

    for (int i = 0; i < 2; i++)
    {
      if (_d_cgr[i] != NULL)
        cudaGraphicsUnregisterResource(_d_cgr[i]);

      glNamedRenderbufferStorage(_rb[i], GL_RGBA8, _width, _height);

      cudaGraphicsGLRegisterImage(&_d_cgr[i], _rb[i], GL_RENDERBUFFER,
        cudaGraphicsRegisterFlagsSurfaceLoadStore
        | cudaGraphicsRegisterFlagsWriteDiscard);
    }

    cudaGraphicsMapResources(2, &_d_cgr[0], 0);
    for (int index = 0; index < 2; index++)
      cudaGraphicsSubResourceGetMappedArray(&_d_ca[index], _d_cgr[index], 0, 0);
    cudaGraphicsUnmapResources(2, &_d_cgr[0], 0);

    return cuda_err;
  }

  void
  Interop::getSize(unsigned int& w, unsigned int& h)
  {
    w = _width;
    h = _height;
  }


} // namespace drive
