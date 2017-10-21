#pragma once

#include "scene.h"

namespace processor
{
  class GPUProcessor
  {
    public:
      GPUProcessor(scene::Scene& scene, int w, int h);

    public:
      bool
      init();

      void
      run();

    private:
      scene::Scene &_scene;

      int _width;
      int _height;
  };
} // namespace processor
