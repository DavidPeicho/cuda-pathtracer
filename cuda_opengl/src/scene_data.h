#pragma once

#include <glm/common.hpp>
#include <tiny_obj_loader.h>

namespace scene
{
  template <typename T>
  struct Buffer
  {
    size_t size;
    T *data;
  };

  struct Mesh
  {
    struct Buffer<tinyobj::index_t> indices;
    struct Buffer<int> material_ids;
  };

  struct SceneData
  {
    struct Buffer<struct Mesh> meshes;
    struct Buffer<tinyobj::real_t> vertices;
    struct Buffer<tinyobj::real_t> normals;
    struct Buffer<struct Material> materials;
    struct Buffer<struct LightProp> lights;
  };

  struct Material
  {
    tinyobj::real_t ambient[3];
    tinyobj::real_t diffuse[3];
    tinyobj::real_t specular[3];
    tinyobj::real_t transmittance[3];
    tinyobj::real_t emission[3];
    tinyobj::real_t shininess;
  };

  struct LightProp
  {
    tinyobj::real_t color[3];
    glm::vec3 vec;
  };

  struct Camera
  {
    glm::ivec2 res;
    glm::vec3 position;
    glm::vec3 dir;
    glm::vec3 u;
    glm::vec3 v;
    float fov_x;
  };

  struct Ray
  {
    glm::vec3 dir;
    glm::vec3 origin;
  };

}
