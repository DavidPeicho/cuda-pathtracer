#pragma once

#include <cuda.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "driver/interop.h"
#include "driver/gpu_info.h"

#include "scene.h"

# define M_PI 3.14159265358979323846

namespace processor
{
  class GPUProcessor
  {
    public:
      GPUProcessor(scene::Scene& scene, int w, int h);
      ~GPUProcessor()
      {
        cudaFree(_d_temporal_framebuffer);
      }

    public:
      bool
      init();

      void
      run(float delta);

      void
      setKeyState(const unsigned int key, bool state);

      inline void
      setMousePos(const double x, const double y)
      {
        double x_step = (x - _width * 0.5);
        double y_step = (y - _height * 0.5);

        _angle.x -= x_step * 0.001;
        _angle.y += y_step * 0.001;

        glm::vec3 offset = glm::vec3(
          cos(_angle.y) * sin(_angle.x),
          sin(_angle.y),
          cos(_angle.y) * cos(_angle.x)
        );

        _scene.getCamPointer()->dir += offset;
        _scene.getCamPointer()->dir = glm::normalize(_scene.getCamPointer()->dir);
      }

      bool
      isKeyPressed(const unsigned int key);

      inline void
      setMoved(bool moved)
      {
        _moved = moved;
      }

      inline driver::Interop&
      getInterop()
      {
        return _interop;
      }

    private:
      scene::Scene &_scene;
      int _width;
      int _height;

      driver::Interop _interop;
      driver::GPUInfo _gpu_info;
      cudaStream_t _stream;

      glm::vec3* _d_temporal_framebuffer;

      bool _keys[65536];
      bool _moved;
      float _cam_speed;

      glm::vec2 _angle = glm::vec2(0.0f, M_PI);
  };
} // namespace processor
