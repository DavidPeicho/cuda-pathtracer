#include <cuda_runtime.h>

#include <math_functions.h>

#include <curand.h>
#include <curand_kernel.h>

#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <stdbool.h>

#include "driver/cuda_helper.h"
#include "utils.h"
#include "scene_data.h"

#include <iostream>

//#define EPSILON 0.0000001;

surface<void, cudaSurfaceType2D> surf;

union rgba_24
{
	uint1 b32;

	struct
	{
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  a : 8;
	};
};

struct IntersectionData
{
  float dist;
  float specular_col;
  glm::vec3 normal;
  glm::vec3 diffuse_col;
  glm::vec3 normal_col;
  const scene::LightProp *light;
  bool is_light;
};

/*
  The sampleTexture() method uses overloading to sample
  several type of texture easily. It allows us to sample
  3-channels texture as well as 1-channel textures.
*/

__device__ inline int
getTextureIdx(const scene::Texture &texture, const glm::vec2& uv)
{
  int x = uv.x * (texture.w - 1);
  int y = uv.y * (texture.h - 1);
  return (y * texture.w + x) * texture.nb_chan;
}

__device__ void
sampleTexture(const scene::Buffer<scene::Texture>& textures,
  int tex_id, const glm::vec2& uv, glm::vec3& out)
{
  const scene::Texture &tex = textures.data[tex_id];
  int idx = getTextureIdx(tex, uv);

  out.x = tex.data[idx];
  out.y = tex.data[idx + 1];
  out.z = tex.data[idx + 2];
}

HOST_DEVICE inline scene::Ray
generateRay(const int x, const int y,
            const int half_w, const int half_h, const scene::Camera &cam)
{
  float screen_dist = half_w / std::tan(cam.fov_x * 0.5);

  scene::Ray ray;
  ray.origin = cam.position;

  glm::vec3 screen_pos = cam.position + (cam.dir * screen_dist)
    + (cam.u * (float)(x - half_w)) + (cam.v * (float)(y - half_h));

  ray.dir = screen_pos - cam.position;
  ray.dir = glm::normalize(ray.dir);
  return ray;
}

__device__ inline bool
intersectTriangle(const glm::vec3 *vert, const glm::vec3 *v_norm,
  const glm::vec2 *v_uv, glm::vec3 &out_normal, glm::vec2 &out_uv,
  const scene::Ray &ray, float& t)
{
	glm::vec3 v0v1 = vert[1] - vert[0];
	glm::vec3 v0v2 = vert[2] - vert[0];
	glm::vec3 p_vec = glm::cross(ray.dir, v0v2);
	float det = glm::dot(v0v1, p_vec);
	if (det < 0.0000001)
		return false;

	float inv_det = __fdividef(1.f, det);
	glm::vec3 t_vec = ray.origin - vert[0];
	float u = glm::dot(t_vec, p_vec) * inv_det;
	if (u < 0 || u > 1) return false;

	glm::vec3 qvec = glm::cross(t_vec, v0v1);
	float v = glm::dot(ray.dir, qvec) * inv_det;
	if (v < 0 || u + v > 1) return false;

  // Interpolates normals
  //out_normal = (1.0f - u - v) * v_norm[0] + u * v_norm[1] + v * v_norm[2];
  out_normal = v_norm[0];
  // Interpolates uvs
  out_uv = (1.0f - u - v) * v_uv[0] + u * v_uv[1] + v * v_uv[2];
  out_uv = glm::mod(out_uv, 1.0f); // UVs should be in [0.0, 1.0]

	t = glm::dot(v0v2, qvec) * inv_det;
	return true;
}

__device__ bool
intersectSphere(const scene::Ray &r, const scene::LightProp & const light, float& t)
{
  static const float epsilon = 0.01f;

	glm::vec3 op = light.vec - r.origin;
	float b = dot(op, r.dir);
	float disc = b*b - dot(op, op) + light.radius * light.radius;
	if (disc < 0.0) return 0;

  disc = __fsqrt_rn(disc);
	(t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);

	return t != 0.0;
}

__device__ inline bool
intersect(const scene::Ray& r,
  const struct scene::SceneData *const scene, IntersectionData &intersection)
{
  static const float MAX_DIST = 100000.0;
  float inter_dist = MAX_DIST;
  intersection.dist = MAX_DIST;

	glm::vec3 vertex[3];
  glm::vec3 v_normal[3];
  glm::vec2 v_uv[3];

  glm::vec3 normal;
  glm::vec2 uv;

  // Checks meshes intersection
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
        v_normal[v].x = scene->normals.data[3 * idx.normal_index];
        v_normal[v].y = scene->normals.data[3 * idx.normal_index + 1];
        v_normal[v].z = scene->normals.data[3 * idx.normal_index + 2];
        v_uv[v].x = scene->texcoords.data[2 * idx.texcoord_index];
        v_uv[v].y = scene->texcoords.data[2 * idx.texcoord_index + 1];
			}
			if (intersectTriangle(vertex, v_normal, v_uv, normal, uv, r, inter_dist)
          && inter_dist < intersection.dist && inter_dist > 0.0)
			{
        const scene::Material& mat = scene->materials.data[mesh.material_ids.data[i / 3]];
        intersection.dist = inter_dist;
        intersection.normal = normal;

        sampleTexture(scene->textures, mat.diffuse_spec_map, uv, intersection.diffuse_col);
        /*if (mat.diffuse_map >= 0)
          sampleTexture(scene->textures, mat.diffuse_map, uv, intersection.diffuse_col);
        else
          intersection.diffuse_col = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);

        if (mat.spec_map >= 0)
          sampleTexture(scene->textures, mat.spec_map, uv, intersection.specular_col);*/

        intersection.is_light = false;
        intersection.light = NULL;
			}
		}
	}

  // At least one intersection has been found.
  for (size_t l = 0; l < scene->lights.size; ++l)
  {
    // Checks lights intersection
    const scene::LightProp &const light = scene->lights.data[l];
    if (intersectSphere(r, light, inter_dist)
        && inter_dist < intersection.dist && inter_dist >= 0.0)
    {
      intersection.is_light = true;
      intersection.light = &light;
      intersection.dist = inter_dist;
      intersection.diffuse_col = glm::vec3(light.color[0], light.color[1], light.color[2]);
      intersection.normal = glm::normalize(light.vec - (inter_dist * r.dir));
    }
  }

	return intersection.dist < MAX_DIST;
}

#define M_PI 3.14159265359f

__device__ inline glm::vec3
brdf_oren_nayar(float n_dot_v, float n_dot_l, glm::vec3 light_dir, glm::vec3 view_dir,
						  glm::vec3 n, float roughness, float metalness, glm::vec3 base_color)
{
	float angle_v_n = acos(n_dot_v);
	float angle_l_n = acos(n_dot_l);

	float alpha = max(angle_v_n, angle_l_n);
	float beta = min(angle_v_n, angle_l_n);
	float gamma = dot(view_dir - n * n_dot_v, light_dir - n * n_dot_l);

	float roughness_2 = roughness * roughness;

	float A = 1.0 - 0.5 * (roughness_2 / (roughness_2 + 0.57));
	float B = 0.45 * (roughness_2 / (roughness_2 + 0.09));
	float C = sin(alpha) * tan(beta);

	float L1 = max(0.0, n_dot_l) * (A + B * max(0.0, gamma) * C);

	glm::vec3 color = base_color;
	color = glm::mix(color, glm::vec3(0.0), metalness);

	return glm::vec3(glm::vec3(color) * glm::vec3(L1));
}

__device__ inline glm::vec3
uncharted_tonemap(glm::vec3 x)
{
   const float A = 0.15;
   const float B = 0.50;
   const float C = 0.10;
   const float D = 0.20;
   const float E = 0.02;
   const float F = 0.30;
   const float W = 11.2;

   return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

__device__ inline glm::vec3
exposure(glm::vec3 color)
{
  const float exposure_bias = 2.0f;
  glm::vec3 curr = uncharted_tonemap(exposure_bias * color);

  const glm::vec3 W = glm::vec3(11.2);
  glm::vec3 white_scale = 1.0f / uncharted_tonemap(W);

  return curr * white_scale;
}

__device__ inline glm::vec3
sample_lights(scene::Ray& r, const scene::Buffer<scene::LightProp>& lights,
  glm::vec3 color, float PDF, glm::vec3 normal)
{
	glm::vec3 L = glm::vec3(0.0f);
	for (int i = 0; i < lights.size; i++)
	{
    const scene::LightProp& const l = lights.data[i];
		float n_dot_l = glm::dot(normal, l.vec - r.origin);
		L += color * n_dot_l * l.emission / PDF;
	}

	return L;
}

__device__ inline glm::vec3 radiance(scene::Ray& r,
  const struct scene::SceneData *const scene, const scene::Camera * const cam,
  curandState* rand_state, int is_static, int static_samples)
{
  glm::vec3 acc = glm::vec3(0.0f, 0.0f, 0.0f);

  // Contains information about each intersection.
  // This will be updated at each call to 'intersect'.
  IntersectionData inter;

  //const int max_bounces = 1 + is_static * (static_samples + 1);
  const int max_bounces = 1;
  for (int b = 0; b < max_bounces; b++)
  {
    glm::vec3 oriented_normal;
    // For energy compensation on Russian roulette
    glm::vec3 thoughput = glm::vec3(1.0f);

    if (intersect(r, scene, inter))
    {
      float cos_theta = glm::dot(inter.normal, r.dir);
      oriented_normal = cos_theta < 0 ? inter.normal : inter.normal * -1.0f;

      //acc += mask * emission * (float)light_emitter * intersection;
      if (inter.is_light)
      {
        acc += inter.light->emission * thoughput;
        continue;
      }

      float r1 = curand_uniform(rand_state);
      float phi = 2.0f * M_PI * curand_uniform(rand_state);

      float sin_t = sqrtf(r1);
      float cos_t = sqrt(1.f - r1);

      glm::vec3 u = glm::normalize(glm::cross(fabs(oriented_normal.x) > .1 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f), oriented_normal));
      glm::vec3 v = glm::cross(oriented_normal, u);

      //Diffuse hemishphere reflection
      glm::vec3 d = glm::normalize(v * sin_t * cos(phi) + u * sin(phi) * sin_t + oriented_normal * cos_t);
      //Specular model (Snell's law)
      //glm::vec3 d = r.dir - 2.0f * oriented_normal * cos_theta;

      // Oren-Nayar diffuse
      //glm::vec3 BRDF = brdf_oren_nayar(cos_theta, n_dot_l, light_dir, r.dir, oriented_normal, 0.1f, 0.99f, color);
      r.origin += r.dir * inter.dist;

      r.origin += oriented_normal * 0.03f;
      r.dir = d;

      //mask *= intersection * color + (1.0f - intersection) * 1.0f;
      //Lambert BRDF/PDF
      //glm::vec3 BRDF = inter.diffuse_col /** n_dot_l*/; // Divided by PI
      glm::vec3 BRDF = inter.diffuse_col /** n_dot_l*/; // Divided by PI
      float PDF = cos_theta; // Divided by PI
                             //glm::vec3 BRDF = color;
      glm::vec3 direct_light = BRDF / PDF;
      thoughput *= direct_light;

      return BRDF;
      acc += thoughput * sample_lights(r, scene->lights, inter.diffuse_col, PDF, inter.normal);

      // Russian roulette
      float p = fmaxf(thoughput.x, fmaxf(thoughput.y, thoughput.z));
      if (r1 > p)
        return acc;

      thoughput *= 1.f / p;
    }
  }

  return acc;
}

__global__ void
kernel(const unsigned int width, const unsigned int height,
	const scene::SceneData *const scene, scene::Camera cam, unsigned int hash_seed,
  int frame_nb, glm::vec3 *temporal_framebuffer, bool moved)
{
  const unsigned int half_w = width / 2;
  const unsigned int half_h = height / 2;

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height) return;

	const unsigned int tid = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	union rgba_24 rgbx;
	rgbx.a = 0.0;

	curandState rand_state;
	curand_init(hash_seed + tid, 0, 0, &rand_state);

	scene::Ray r = generateRay(x, y, half_w, half_h, cam);

	int is_static = !moved;
	int static_samples = 1;
	int samples = 2 + is_static * static_samples;

  glm::vec3 rad = glm::vec3(0.0f);
	for (int i = 0; i < samples; i++)
		rad += radiance(r, scene, &cam, &rand_state, is_static, static_samples);
  rad /= samples;

	rad = glm::clamp(rad, 0.0f, 1.0f);

	int i = (height - y - 1) * width + x;
	temporal_framebuffer[i] *= is_static;
	temporal_framebuffer[i] += rad;

	rad = temporal_framebuffer[i] / (float)frame_nb;

	rad = exposure(rad);
	rad = glm::pow(rad, glm::vec3(1.0f / 2.2f));

    rgbx.r = rad.x * 255;
    rgbx.g = rad.y * 255;
    rgbx.b = rad.z * 255;

	surf2Dwrite(rgbx.b32,
		surf,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero);
}

inline unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);

	return a;
}

cudaError_t
raytrace(cudaArray_const_t array, const scene::SceneData *const scene, const scene::Camera * const cam,
	const unsigned int width, const unsigned int height, cudaStream_t stream,
	glm::vec3 *temporal_framebuffer, bool moved)
{
	static unsigned int seed = 0;

	if (moved)
		seed = 0;

	seed++;

	cudaBindSurfaceToArray(surf, array);

	// Register occupancy : nb_threads = regs_per_block / 32
	// Shared memory occupancy : nb_threads = shared_mem / 32
	// Block size occupancy

	// TODO: We should get into account GPU info, such as number of registers,
	// shared memory size, warp size, etc...
	dim3 threads_per_block(16, 16);
	dim3 nb_blocks(width / threads_per_block.x, height / threads_per_block.y);

	if (nb_blocks.x > 0 && nb_blocks.y > 0)
		kernel << <nb_blocks, threads_per_block, 0, stream >> > (width, height, scene, *cam,
      WangHash(seed), seed, temporal_framebuffer, moved);

	return cudaSuccess;
}
