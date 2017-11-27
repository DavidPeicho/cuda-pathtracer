#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

struct __align__(8) IntersectionData
{
  float3 normal;
  float3 surface_normal;
  float3 tangent;
  float3 diffuse_col;
  float2 uv;
  const scene::LightProp* light;
  float dist;
  float specular_col;
  float ior;
};

__device__ inline int
getTextureIdx(const scene::Texture& texture, const float2& uv)
{
  int x = uv.x * (texture.w - 1);
  int y = uv.y * (texture.h - 1);
  return (y * texture.w + x) * texture.nb_chan;
}

/// <summary>
/// Fetch a given texture, using interpolated UVs.
/// </summary>
/// <param name="textures">Textures list.</param>
/// <param name="tex_id">ID of the texture to fetch.</param>
/// <param name="uv">Interpolated UVs used for the fetch.</param>
/// <param name="out">Contains the texture fetch value, as a float3.</param>
__device__ inline void
sampleTexture(const scene::Buffer<scene::Texture>& textures, int tex_id,
              const float2& uv, float3& out)
{
  const scene::Texture& tex = textures.data[tex_id];
  int idx = getTextureIdx(tex, uv);

  out.x = tex.data[idx];
  out.y = tex.data[idx + 1];
  out.z = tex.data[idx + 2];
}

/// <summary>
/// Fetch a given texture, using interpolated UVs.
/// </summary>
/// <param name="textures">Textures list.</param>
/// <param name="tex_id">ID of the texture to fetch.</param>
/// <param name="uv">Interpolated UVs used for the fetch.</param>
/// <param name="out">Contains the texture fetch value, as a float4.</param>
__device__ inline void
sampleTexture(const scene::Buffer<scene::Texture>& textures, int tex_id,
              const float2& uv, float4& out)
{
  const scene::Texture& tex = textures.data[tex_id];
  int idx = getTextureIdx(tex, uv);

  out.x = tex.data[idx];
  out.y = tex.data[idx + 1];
  out.z = tex.data[idx + 2];
  out.w = tex.data[idx + 3];
}

/// <summary>
/// Generates a ray from the camera.
/// </summary>
/// <param name="x">Horizontal coordinate on the virtual screen.</param>
/// <param name="y">Vertical coordinate on the virtual screen.</param>
/// <param name="half_w">Half of the width of the virtual screen.</param>
/// <param name="half_h">Half of the height of the virtual screen.</param>
/// <param name="cam">Scene camera.</param>
HOST_DEVICE inline scene::Ray
generateRay(const int x, const int y, const int half_w, const int half_h,
            scene::Camera& cam)
{
  float screen_dist = half_w / tanf(cam.fov_x * 0.5f);

  scene::Ray ray;
  ray.origin = cam.position;

  cam.u = normalize(cross(cam.dir, make_float3(0.0, -1.0, 0.0)));
  cam.v = normalize(cross(cam.u, cam.dir));

  cam.u *= -1.0;

  float3 screen_pos = cam.position + (cam.dir * screen_dist) +
                      (cam.u * (float)(x - half_w)) +
                      (cam.v * (float)(y - half_h));

  ray.dir = screen_pos - cam.position;
  ray.dir = normalize(ray.dir);

  return ray;
}

/// <summary>
/// Checks a ray-triangle intersection.
/// </summary>
__device__ inline bool
intersectTriangle(const scene::Face& face, float3& out_normal, float2& out_uv,
                  const scene::Ray& ray, float& t)
{
  float3 v0v1 = face.vertices[1] - face.vertices[0];
  float3 v0v2 = face.vertices[2] - face.vertices[0];
  float3 p_vec = cross(ray.dir, v0v2);
  float det = dot(v0v1, p_vec);
  if (det < 0.0000001)
    return false;

  float inv_det = __fdividef(1.f, det);
  float3 t_vec = ray.origin - face.vertices[0];
  float u = dot(t_vec, p_vec) * inv_det;
  if (u < 0 || u > 1)
    return false;

  float3 qvec = cross(t_vec, v0v1);
  float v = dot(ray.dir, qvec) * inv_det;
  if (v < 0 || u + v > 1)
    return false;

  // Interpolates normals
  out_normal = (1.0f - u - v) * face.normals[0] + u * face.normals[1] +
               v * face.normals[2];
  // out_normal = face.normals[0];
  // Interpolates uvs
  out_uv = (1.0f - u - v) * face.texcoords[0] + u * face.texcoords[1] +
           v * face.texcoords[2];
  out_uv = mod(out_uv, 1.0);

  t = dot(v0v2, qvec) * inv_det;
  return true;
}

/// <summary>
/// Checks a ray-sphere intersection, using a parametric equation.
/// </summary>
__device__ bool
intersectSphere(const scene::Ray& r, const scene::LightProp& light, float& t)
{
  static const float epsilon = 0.01f;

  float3 op = light.vec - r.origin;
  float b = dot(op, r.dir);
  float disc = b * b - dot(op, op) + light.radius * light.radius;
  if (disc < 0.0)
    return 0;

  disc = __fsqrt_rn(disc);
  (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);

  return t != 0.0;
}

/// <summary>
/// Checks an intersection between a ray and all the meshes of the
/// scene pointed by `scene_id'.
/// </summary>
__device__ inline bool
intersect(const scene::Ray& r, const scene::Scenes* scenes,
          unsigned int scene_id, IntersectionData& intersection)
{
  static const float MAX_DIST = 100000.0;

  const scene::SceneData* scene = scenes->scenes[scene_id];
  const scene::Buffer<scene::Texture>& textures = scenes->textures;

  float inter_dist = MAX_DIST;
  intersection.dist = MAX_DIST;

  float3 normal;
  float2 uv;

  const scene::Material* inter_mat = nullptr;

  // Checks meshes intersection
  for (size_t m = 0; m < scene->meshes.size; ++m) {
    const auto& mesh = scene->meshes.data[m];
    for (size_t i = 0; i < mesh.faces.size; ++i) {
      const auto& face = mesh.faces.data[i];
      if (intersectTriangle(face, normal, uv, r, inter_dist) &&
          inter_dist < intersection.dist && inter_dist > 0.0) {
        inter_mat = &scene->materials.data[face.material_id];
        intersection.ior = inter_mat->ior;
        intersection.normal = normal;
        intersection.surface_normal = normal;
        intersection.tangent = face.tangent;

        intersection.uv = uv;
        intersection.dist = inter_dist;
        intersection.light = NULL;
      }
    }
  }

  // At least one intersection has been found.
  for (unsigned l = 0; l < scene->lights.size; ++l) {
    // Checks lights intersection
    const scene::LightProp& light = scene->lights.data[l];
    if (intersectSphere(r, light, inter_dist) &&
        inter_dist < intersection.dist && inter_dist >= 0.0) {
      intersection.light = &light;
      intersection.dist = inter_dist;
      intersection.diffuse_col =
        make_float3(light.color.x, light.color.y, light.color.z);
      intersection.normal = normalize(light.vec - (inter_dist * r.dir));
      intersection.surface_normal = intersection.surface_normal;
      inter_mat = nullptr;
    }
  }

  // Samples texture & computes normal mapping only once,
  // when we are sure this is the closest intersection.
  if (inter_mat) {
    // Fetches diffuse color from texture
    float4 fetch;
    sampleTexture(textures, inter_mat->diffuse_spec_map, intersection.uv,
                  fetch);
    intersection.diffuse_col.x = fetch.x;
    intersection.diffuse_col.y = fetch.y;
    intersection.diffuse_col.z = fetch.z;

    intersection.specular_col = fetch.w;

    // Computes normal perturbated by normal map
    if (inter_mat->normal_map >= 0) {
      float3 normal;
      sampleTexture(textures, inter_mat->normal_map, intersection.uv, normal);
      intersection.normal = normalize((normal * 2.0f) - 1.0f);

      float3 binormal =
        normalize(cross(intersection.tangent, intersection.surface_normal));

      mat3 tbn;
      tbn.x = intersection.tangent;
      tbn.y = -binormal;
      tbn.z = intersection.surface_normal;

      intersection.normal = tbn * intersection.normal;
    }
  }

  return intersection.dist < MAX_DIST;
}