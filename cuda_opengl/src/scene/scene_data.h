#pragma once

#include <cuda_runtime.h>
#include <tiny_obj_loader.h>

namespace scene
{
  template <typename T>
  struct __align__(16) Buffer
  {
    unsigned int size;
    T *data;
  };

  struct __align__(8) Cubemap
  {
    cudaArray *cubemap;
    cudaChannelFormatDesc cubemap_desc;
  };

  struct __align__(16) Texture
  {
    int w;
    int h;
    int nb_chan;
    float *data;
  };

  struct __align__(16) Face
  {
    float3 vertices[3];
    float3 normals[3];
    float2 texcoords[3];
    float3 tangent;
    unsigned int material_id;
  };

  struct __align__(16) Mesh
  {
    struct Buffer<Face> faces;
  };

  struct __align__(16) SceneData
  {
    struct Buffer<Mesh> meshes;
    struct Buffer<struct Material> materials;
    struct Buffer<struct LightProp> lights;
    struct Buffer<struct Texture> textures;
  };

  struct __align__(8) Material
  {
    int diffuse_spec_map;
    int normal_map;
	  float ior;
  };

  struct __align__(8) LightProp
  {
    float3 color;
    float3 vec;
    float emission;
    float radius;
  };

  struct __align__(8) Camera
  {
    float3 position;
    float3 dir;
    float3 u;
    float3 v;
    float fov_x;
    float speed;
  };

  struct __align__(16) Ray
  {
    float3 dir;
    float3 origin;
  };

}
