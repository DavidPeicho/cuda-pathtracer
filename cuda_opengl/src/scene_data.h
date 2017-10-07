#pragma once

#include <tiny_obj_loader.h>

struct Mesh
{
  tinyobj::index_t *indices;
  size_t nb_indices;
  int *material_ids;
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
