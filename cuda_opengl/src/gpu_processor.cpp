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

  GPUProcessor::GPUProcessor(std::vector<std::string> scene_names, int width, int height)
               : _scene_names(scene_names)
    , _scene_id(0)
    , _prev_scene_id(0)
               , _interop(width, height)
               , _d_temporal_framebuffer(nullptr)
               , _moved(false)
  {
    cudaStreamCreateWithFlags(&_stream, cudaStreamDefault);
    cudaMalloc(&_d_temporal_framebuffer, width * height * sizeof(float3));

    // Initializes all keys to released
    for (size_t i = 0; i < 65536; ++i) _keys[i] = false;

    // Creates associated files scenes
    for (const auto &file : scene_names) _scenes.emplace_back(file);
  }

  void
  GPUProcessor::init()
  {
    std::cout << "Uploading scenes to VRAM...\n"
              << "This operation may take a few seconds.\n" << std::endl;

    // Logs information about the GPUs.
    std::cout << _gpu_info.getProfile() << std::endl;

    // Uploads scene for GPU usage
    for (auto& scene : _scenes) scene.upload(&_camera);

    std::cout << "Uploading has terminated!" << std::endl;
    // Logs information about the GPUs.
    std::cout << _gpu_info.getProfile() << std::endl;
  }

  void
  GPUProcessor::update(float delta)
  {
    // Updates cam rotation
    _camera.u = cross(WORLD_DOWN_VEC, _camera.dir);
    _camera.v = cross(_camera.dir, _camera.u);

    // Updates cam position
    if (this->isKeyPressed(GLFW_KEY_W))
      _camera.position += _camera.dir * _actual_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_S))
      _camera.position -= _camera.dir * _actual_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_A))
      _camera.position -= _camera.u * _actual_speed * delta;
    if (this->isKeyPressed(GLFW_KEY_D))
      _camera.position += _camera.u * _actual_speed * delta;

    if (this->isKeyPressed(GLFW_KEY_LEFT_SHIFT))
      _actual_speed += 0.05;
    else
      _actual_speed = _camera.speed;
  }

  void
  GPUProcessor::render()
  {
    if (_scenes.size() == 0) return;

    if (_prev_scene_id != _scene_id)
    {
      cudaDeviceSynchronize();
      _prev_scene_id = _scene_id;
      std::cout << "toto" << std::endl;
    }

    const auto *cpu_scene = _scenes[_scene_id].getScenePointer();
    const auto *gpu_scene = _scenes[_scene_id].getUploadedScenePointer();

    if (_scene_id == 1)
      //std::cout << "toto" << std::endl;

    if (gpu_scene == nullptr || cpu_scene == nullptr) return;

    cudaError_t cuda_err = _interop.map(_stream);
    raytrace(
      _interop.getArray(), cpu_scene, gpu_scene, &_camera,
      _interop.width(), _interop.height(), _stream,
      _d_temporal_framebuffer, _moved
    );
    cuda_err = _interop.unmap(_stream);

    this->setMoved(false);

    _interop.blit();
    _interop.swap();
  }

  void
  GPUProcessor::resize(unsigned int w, unsigned int h)
  {
    _interop.setSize(w, h);

    if (_d_temporal_framebuffer != nullptr)
      cudaFree(_d_temporal_framebuffer);

    cudaMalloc(&_d_temporal_framebuffer, h * w * sizeof(float3));
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
