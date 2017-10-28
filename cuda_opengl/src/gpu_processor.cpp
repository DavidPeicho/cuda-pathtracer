#include <cuda.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW\glfw3.h>
#include "gpu_processor.h"
#include "raytrace.h"

namespace processor
{
  GPUProcessor::GPUProcessor(scene::Scene& scene, int width, int height)
               : _scene(scene)
               , _width(width)
               , _height(height)
               , _interop(width, height)
  {
    cudaSetDevice(_gpu_info.getCUDAGPU().device_id);
    // Logs information about the GPUs.
    std::cout << _gpu_info.getProfile() << std::endl;

    _scene.upload(false);

    // Logs information about the GPUs, allows to see
    // how much memory is consummed by the obj scene.
    std::cout << _gpu_info.getProfile() << std::endl;

    cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);

    cudaMalloc(&_d_temporal_framebuffer, _width * _height * sizeof(glm::vec3));
  }

  bool
  GPUProcessor::init()
  {
    // Uploads scene for GPU usage
    _scene.upload(false);
    return _scene.ready();
  }

  void
  GPUProcessor::run()
  {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int pow2_width = 0;
    unsigned int pow2_height = 0;
    _interop.getSize(pow2_width, pow2_height);
    cudaError_t cuda_err = _interop.map(_stream);

    cuda_err = _interop.map(_stream);
    cuda_err = raytrace(_interop.getArray(), _scene.getSceneData(), _width,
      _height, _stream, offset, dir_offset, _d_temporal_framebuffer, _moved);
    cuda_err = _interop.unmap(_stream);

    this->setMoved(false);

    if (this->isKeyPressed(GLFW_KEY_Z))
    offset.z++;
    if (this->isKeyPressed(GLFW_KEY_S))
    offset.z--;
    if (this->isKeyPressed(GLFW_KEY_D))
    offset.x++;
    if (this->isKeyPressed(GLFW_KEY_Q))
    offset.x--;
    if (this->isKeyPressed(GLFW_KEY_SPACE))
    offset.y++;
    if (this->isKeyPressed(GLFW_KEY_LEFT_CONTROL))
    offset.y--;

    _interop.blit();
    _interop.swap();
  }

  void
  GPUProcessor::setKeyState(const unsigned int key, bool state)
  {
    _keys[key] = state;
  }

  bool
  GPUProcessor::isKeyPressed(const unsigned int key)
  {
    bool k = _keys[key];
    if (k)
      this->setMoved(true);
    return k;
  }

} // namespace processor