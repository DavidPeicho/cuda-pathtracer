#pragma once

#include <cuda_runtime.h>
#include <tiny_obj_loader.h>

namespace scene {
/// <summary>
/// Templated GPU-aligned array.
/// </summary>
template <typename T>
struct __align__(16) Buffer
{
  unsigned int size;
  T* data = nullptr;
};

/// <summary>
/// GPU-aligned Cubemap containing the pixel data, as well as
/// the Cubemap format (number of channels, etc...)
/// </summary>
struct __align__(8) Cubemap
{
  cudaArray* cubemap;
  cudaChannelFormatDesc cubemap_desc;
};

/// <summary>
/// GPU-aligned Textures with is size, number of channels,
/// and raw data.
/// </summary>
struct __align__(16) Texture
{
  int w;
  int h;
  int nb_chan;
  float* data;
};

/// <summary>
/// GPU-aligned primitive containing
/// * The three vertices making the face;
/// * One normal per vertex;
/// * One tangent for the whole primitive;
/// * The id of the material shading this primitive.
/// </summary>
struct __align__(16) Face
{
  float3 vertices[3];
  float3 normals[3];
  float2 texcoords[3];
  float3 tangent;
  unsigned int material_id;
};

/// <summary>
/// GPU-aligned Mesh.
/// A mesh is simply described by a list of faces.
/// </summary>
struct __align__(16) Mesh
{
  struct Buffer<Face> faces;
};

/// <summary>
/// GPU-aligned SceneData.
/// SceneData contains data relative to only one scene:
/// * meshes: list of meshes;
/// * materials: list of materials;
/// * lights: list of lights.
/// </summary>
struct __align__(16) SceneData
{
  struct Buffer<Mesh> meshes;
  struct Buffer<struct Material> materials;
  struct Buffer<struct LightProp> lights;
};

/// <summary>
/// GPU-aligned Scenes.
/// Contains the list of all tge scenes, as well
/// as all the textures.
/// </summary>
struct __align__(16) Scenes
{
  struct Buffer<struct Texture> textures;
  struct SceneData** scenes;
};

/// <summary>
/// GPU-aligned Material.
/// * diffuse_spec_map: id of the diffuse texture (color);
/// * normal_map: id of the normal map texture;
/// * ior: index of refraction.
/// </summary>
struct __align__(8) Material
{
  int diffuse_spec_map;
  int normal_map;
  float ior;
};

/// <summary>
/// GPU-aligned LightProp.
/// * color: color of the light;
/// * vec: position of the light;
/// * emission: emmission factor;
/// * radius: size of the sphere radius.
/// </summary>
struct __align__(8) LightProp
{
  float3 color;
  float3 vec;
  float emission;
  float radius;
};

/// <summary>
/// GPU-aligned Camera.
/// Contains the camera data. This struct will be passed
/// to the kernel directly on the stack for performance
/// reasons.
/// </summary>
struct __align__(8) Camera
{
  float3 position;
  float3 dir;
  float3 u;
  float3 v;
  float fov_x;
  float speed;
  float aperture;
  float focus_dist;
};

/// <summary>
/// GPU-aligned Ray.
/// dir: direction of the ray;
/// origin: origin of the ray.
/// </summary>
struct __align__(16) Ray
{
  float3 dir;
  float3 origin;
};
}
