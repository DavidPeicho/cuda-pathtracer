#pragma once

//#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <scene/scene_data.h>

namespace gui
{
  /// <summary>
  /// Singleton of the UI. This can be use everywhere
  /// in the code.
  /// </summary>
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
      postProcess(int& post_id, const std::vector<std::string>& items);

      void
      camera(scene::Camera& cam, float h_offset = 0.0f);

    private:
      static GUIManager *_instance;

    private:
      bool _show;
  };
} // namespace gui
