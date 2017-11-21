#pragma once

#include <cuda.h>
#include <vector>

#include "driver/interop.h"
#include "driver/gpu_info.h"

#include "shaders/cutils_math.h"

#include "scene/scene.h"

# define M_PI 3.14159265358979323846

namespace processor
{
  class GPUProcessor
  {
    public:
      GPUProcessor(std::vector<std::string> scene_names, int w, int h);
      ~GPUProcessor()
      {
        cudaFree(_d_temporal_framebuffer);
      }

    public:
      void
      init();

      void
      update(float delta);

      void
      render();

      void
      resize(unsigned int w, unsigned int h);

      void
      setKeyState(const unsigned int key, bool state);

      inline void
      setMousePos(const double x, const double y)
      {
        double x_step = (x - _interop.half_width());
        double y_step = (y - _interop.half_height());

        _angle.x -= x_step * 0.001;
        _angle.y += y_step * 0.001;

        float3 offset = make_float3(
          cos(_angle.y) * sin(_angle.x),
          sin(_angle.y),
          cos(_angle.y) * cos(_angle.x)
        );

        _camera.dir += offset;
        _camera.dir = normalize(_camera.dir);
      }

      bool
      isKeyPressed(const unsigned int key);

    public:
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

      inline scene::Camera&
      getCamera()
      {
        return _camera;
      }

      inline const std::vector<std::string>&
      getSceneItems()
      {
        return _scene_names;
      }

      inline int&
      getSceneId()
      {
        return _scene_id;
      }

    private:
      std::vector<std::string> _scene_names;
      std::vector<scene::Scene> _scenes;
      int _scene_id;
      int _prev_scene_id;

      scene::Camera _camera;

      driver::Interop _interop;
      driver::GPUInfo _gpu_info;
      cudaStream_t _stream;

      float3* _d_temporal_framebuffer;

      bool _keys[65536];
      bool _moved;
      float _actual_speed;

      float2 _angle = make_float2(0.0f, M_PI);
  };
} // namespace processor
