#pragma once

#include "../scene/scene_data.h"

namespace gui
{
  class GUIManager
  {
    public:
      static inline GUIManager*
      inst()
      {
        if (_instance == nullptr) _instance = new GUIManager();

        return _instance;
      };

    public:
      GUIManager();

    public:
      void
      init(GLFWwindow* window);

      void
      begin();

      void
      render();

      void
      release();

    public:
      float
      info(int &scene_id, int& cubemap_id, const std::vector<std::string>& items,
        const std::vector<std::string>& cubemaps);

      void
      camera(scene::Camera& cam, float h_offset = 0.0f);

    private:
      static GUIManager *_instance;

    private:
      bool _show;
  };
} // namespace gui
