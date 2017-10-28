#pragma once

#include <string>

#include "scene_data.h"

namespace scene
{
  class Scene
  {
    public:
      Scene(const std::string& filepath);
      Scene(const std::string&& filepath);

    public:
      void
      upload(bool is_cpu);

      void
      release(bool is_cpu);

      const inline struct SceneData *
      getSceneData()
      {
        return _scene_data;
      }

      inline bool
      ready()
      {
        return _ready;
      }

      inline std::string&
      error()
      {
        return _load_error;
      }

    private:
      void
      init(const char* filepath);

      void
      upload_cpu(const std::vector<tinyobj::shape_t> &shapes,
        const std::vector<tinyobj::material_t>& materials,
        const tinyobj::attrib_t attrib);

      void
      upload_gpu(const std::vector<tinyobj::shape_t> &shapes,
        const std::vector<tinyobj::material_t>& materials,
        const tinyobj::attrib_t attrib);

      void
      release_cpu();

      void
      release_gpu();

    private:
      std::string _filepath;
      bool _uploaded;
      bool _ready;
      std::string _load_error;

      struct SceneData *_scene_data;
  };

} // namespace scene
