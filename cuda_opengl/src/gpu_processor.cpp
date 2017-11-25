#include <cuda.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "driver/cuda_helper.h"

#include "gpu_processor.h"
#include "shaders/raytrace.h"

namespace processor
{
  namespace
  {
    static const float3 WORLD_DOWN_VEC = make_float3(0.0f, -1.0f, 0.0f);
  }

  GPUProcessor::GPUProcessor(scene::Scene& scene, int width, int height)
               : _scene(scene)
               , _width(width)
               , _height(height)
               , _interop(width, height)
               , _d_temporal_framebuffer(nullptr)
               , _moved(false)
               , _cam_speed(1.0f)
  {
    //cudaSetDevice(_gpu_info.getCUDAGPU().device_id);
    // Logs information about the GPUs.
    std::cout << _gpu_info.getProfile() << std::endl;

    // Logs information about the GPUs, allows to see
    // how much memory is consummed by the obj scene.
    std::cout << _gpu_info.getProfile() << std::endl;

    cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);
    cudaMalloc(&_d_temporal_framebuffer, _width * _height * sizeof(float3));

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

    const auto *cpu_scene = _scene.getScenePointer();
    const auto *gpu_scene = _scene.getUploadedScenePointer();

    raytrace(_interop.getArray(), cpu_scene, gpu_scene, cam, _width,
        _height, _stream, _d_temporal_framebuffer, _moved);

    cuda_err = _interop.unmap(_stream);

    this->setMoved(false);

    // Updates cam rotation
    cam->u = cross(WORLD_DOWN_VEC, cam->dir);
    cam->v = cross(cam->dir, cam->u);

    // Updates cam position
    if (this->isKeyPressed(GLFW_KEY_W)) cam->position += cam->dir * _cam_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_S)) cam->position -= cam->dir * _cam_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_A)) cam->position -= cam->u * _cam_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_D)) cam->position += cam->u * _cam_speed * delta;

    // Resets camera orientation
    if (this->isKeyPressed(GLFW_KEY_SPACE))
    {
      std::cout << cam->dir.x << " | " << cam->dir.y << " | " << cam->dir.z << std::endl;

      cam->dir.x = 0.0f;
      cam->dir.y = 0.0f;
      cam->dir.z = -1.0f;
      _angle.x = 0.0f;
      _angle.y = M_PI;
    }

    if (this->isKeyPressed(GLFW_KEY_LEFT_SHIFT))
    {
      std::cout << cam->position.x << " | " << cam->position.y << " | " << cam->position.z << std::endl;
      _cam_speed += 0.05f;
    }
    else
      _cam_speed = 1.0;

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
