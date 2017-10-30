#pragma once

#include <cuda.h>
#include <glm/glm.hpp>

#include "driver/interop.h"
#include "driver/gpu_info.h"

#include "scene.h"

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
        angle.x -= 0.003f * float(_width / 2 - x);
        angle.y -= 0.003f * float(_height / 2 - y);

        dir_offset = glm::vec3(
          cos(angle.y) * sin(angle.x),
          sin(angle.y),
          cos(angle.y) * cos(angle.x)
        );

        dir_offset = glm::normalize(dir_offset);
      }

      bool
      isKeyPressed(const unsigned int key);

      inline void
      setMoved(bool moved)
      {
        _moved = moved;
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

      // TODO: Refactors this, which is hardcoded for now
      glm::vec3 offset = glm::vec3(0.0f);
      glm::vec2 angle = glm::vec2(0.0f);
      glm::vec3 dir_offset = glm::vec3(0.0f);
  };
} // namespace processor
