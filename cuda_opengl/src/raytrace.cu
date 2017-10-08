#include <cuda_runtime.h>
#include <glm/common.hpp>
#include <glm/glm.hpp>
#include <stdbool.h>

#include "driver/cuda_helper.h"
#include "utils.h"
#include "scene_data.h"

//#define EPSILON 0.0000001;

surface<void,cudaSurfaceType2D> surf;

union rgba_24
{
  uint1 b32;

  struct
  {
    unsigned  r  : 8;
    unsigned  g  : 8;
    unsigned  b  : 8;
    unsigned  a : 8;
  };
};

HOST_DEVICE bool
intersectTriangle(const glm::vec3 *vert, const scene::Ray &ray)
{
  glm::vec3 v0v1 = vert[1] - vert[0];
  glm::vec3 v0v2 = vert[2] - vert[0];
  glm::vec3 p_vec = glm::cross(ray.dir, v0v2);
  double det = glm::dot(v0v1, p_vec);
  if (det > - 0.0000001 && det < 0.0000001)
    return false;

  double inv_det = 1 / det;
  glm::vec3 t_vec = ray.origin - vert[0];
  double u = glm::dot(t_vec, p_vec) * inv_det;
  if (u < 0 || u > 1) return false;

  glm::vec3 qvec = glm::cross(t_vec, v0v1);
  double v = glm::dot(ray.dir, qvec) * inv_det;
  if (v < 0 || u + v > 1) return false;

  double t = glm::dot(v0v2, qvec) * inv_det;
  return true;
}

HOST_DEVICE bool
intersect(const scene::Ray& r,
  const struct scene::SceneData *const scene)
{
  glm::vec3 vertex[3];
  for (size_t m = 0; m < scene->meshes.size; ++m)
  {
    const scene::Mesh &mesh = scene->meshes.data[m];
    for (size_t i = 0; i < mesh.indices.size; i += 3)
    {
      for (size_t v = 0; v < 3; ++v)
      {
        tinyobj::index_t idx = mesh.indices.data[i + v];
        vertex[v].x = scene->vertices.data[3 * idx.vertex_index];
        vertex[v].y = scene->vertices.data[3 * idx.vertex_index + 1];
        vertex[v].z = scene->vertices.data[3 * idx.vertex_index + 2];
      }
      if (intersectTriangle(vertex, r))
        return true;
    }
  }
  return false;
}

__global__ void
kernel(const unsigned int width, const unsigned int height,
  const unsigned int half_w, const unsigned int half_h,
  const scene::SceneData *const scene)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  union rgba_24 rgbx;
  rgbx.r = 0;
  rgbx.g = 0;
  rgbx.b = 0;
  rgbx.a = 0;

  struct scene::Camera *cam = scene->cam;
  double screen_dist = half_w / std::tan(cam->fov_x);

  scene::Ray r;
  r.origin = scene->cam->position;
  r.dir.z = screen_dist;
  r.dir.x = x - half_w;
  r.dir.y = y - half_h;
  r.dir = glm::normalize(r.dir);

  if (intersect(r, scene))
    rgbx.r = 255;

  surf2Dwrite(rgbx.b32,
    surf,
    x * sizeof(rgbx),
    y,
    cudaBoundaryModeZero);
}

cudaError_t
raytrace(cudaArray_const_t array, const scene::SceneData *const scene,
  const unsigned int width, const unsigned int height, cudaStream_t stream)
{
  cudaBindSurfaceToArray(surf, array);

  // Register occupancy : nb_threads = regs_per_block / 32
  // Shared memory occupancy : nb_threads = shared_mem / 32
  // Block size occupancy 

  // TODO: We should get into account GPU info, such as number of registers,
  // shared memory size, warp size, etc...
  dim3 threads_per_block(8, 8);
  dim3 nb_blocks(width / 8, height / 8);

  if (nb_blocks.x > 0 && nb_blocks.y > 0)
    kernel<<<nb_blocks, threads_per_block, 0, stream>>>(width, height,
      width / 2, height / 2, scene);
  
  return cudaSuccess;
}