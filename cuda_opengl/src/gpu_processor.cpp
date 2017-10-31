#include <cuda.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW\glfw3.h>

#include "driver/cuda_helper.h"

#include "gpu_processor.h"
#include "raytrace.h"

namespace processor
{
  namespace
  {
    static const glm::vec3 WORLD_DOWN_VEC(0.0, -1.0, 0.0);
  }

  GPUProcessor::GPUProcessor(scene::Scene& scene, int width, int height)
               : _scene(scene)
               , _width(width)
               , _height(height)
               , _interop(width, height)
  {
    //cudaSetDevice(_gpu_info.getCUDAGPU().device_id);
    // Logs information about the GPUs.
    std::cout << _gpu_info.getProfile() << std::endl;

    // Logs information about the GPUs, allows to see
    // how much memory is consummed by the obj scene.
    std::cout << _gpu_info.getProfile() << std::endl;

    cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);
    cudaMalloc(&_d_temporal_framebuffer, _width * _height * sizeof(glm::vec3));

    // Initializes all keys to released
    for (size_t i = 0; i < 65536; ++i)
      _keys[i] = false;
  }

  bool
  GPUProcessor::init()
  {
    // Uploads scene for GPU usage
    _scene.upload(false);
    return _scene.ready();
  }

  void
  GPUProcessor::run(float delta)
  {
    unsigned int pow2_width = 0;
    unsigned int pow2_height = 0;
    _interop.getSize(pow2_width, pow2_height);

    auto *cam = _scene.getCamPointer();
    cudaError_t cuda_err = _interop.map(_stream);
    raytrace(_interop.getArray(), _scene.getUploadedScenePointer(), cam, _width,
        _height, _stream, _d_temporal_framebuffer, _moved);
    cuda_err = _interop.unmap(_stream);

    this->setMoved(false);

    // Updates cam rotation
    cam->u = cross(WORLD_DOWN_VEC, cam->dir);
    cam->v = cross(cam->dir, cam->u);

    // Updates cam position
    if (this->isKeyPressed(GLFW_KEY_W)) cam->position += cam->dir * delta;
    if (this->isKeyPressed(GLFW_KEY_S)) cam->position -= cam->dir * delta;
    if (this->isKeyPressed(GLFW_KEY_A)) cam->position -= cam->u * delta;
    if (this->isKeyPressed(GLFW_KEY_D)) cam->position += cam->u * delta;

    // Resets camera orientation
    if (this->isKeyPressed(GLFW_KEY_SPACE))
    {
      cam->dir.x = 0.0;
      cam->dir.y = 0.0;
      cam->dir.z = -1.0;
      _angle.x = 0.0;
      _angle.y = M_PI;
    }

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
    if (k) this->setMoved(true);
    return k;
  }

} // namespace processor