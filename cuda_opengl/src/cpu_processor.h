#pragma once

#include <glad/glad.h>

#include "scene.h"

namespace processor
{
  class CPUProcessor
  {
    public:
      CPUProcessor(scene::Scene& scene, int w, int h);
      ~CPUProcessor();

    public:
      bool
      init();

      void
      run();

    private:
      scene::Scene &_scene;
      int _width;
      int _height;

      unsigned int _pbo_idx;
      unsigned int _pbo_next_idx;

      GLuint _pbo[2];
      GLuint _screen_tex;

      GLuint _shader;
      GLuint _quadVbo;
      GLuint _quadEbo;
      GLuint _quadVao;
  };
}  // namespace processor
