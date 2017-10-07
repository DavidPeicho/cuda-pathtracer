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
      ~Scene();

    public:
      void
      upload();
      
      void
      release();

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

    private:
      bool _uploaded;
      bool _ready;
      std::string _load_error;

      std::vector<tinyobj::shape_t> _shapes;
      std::vector<tinyobj::material_t> _materials;
      tinyobj::attrib_t _attrib;

      tinyobj::real_t *_d_vertices;
      tinyobj::real_t *_d_normals;
      Mesh *_d_meshes;
      Material *_d_materials;
    
  };
} // namespace scene
