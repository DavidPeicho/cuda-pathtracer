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

  struct Texture
  {
    int w;
    int h;
    int nb_chan;
    float *data;
  };

  struct Face
  {
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 texcoords[3];
    glm::vec3 tangent;
    int material_id;
  };

  struct Mesh
  {
    struct Buffer<Face> faces;
  };

  struct SceneData
  {
    struct Buffer<Mesh> meshes;
    struct Buffer<struct Material> materials;
    struct Buffer<struct LightProp> lights;
    struct Buffer<struct Texture> textures;
  };

  struct Material
  {
    int diffuse_spec_map;
    int normal_map;
  };

  struct LightProp
  {
    tinyobj::real_t color[3];
    glm::vec3 vec;
    float emission;
    float radius;
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
