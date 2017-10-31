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
intersectTriangle(const glm::vec3 *vert, const scene::Ray &ray, float& t)
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

	t = glm::dot(v0v2, qvec) * inv_det;
	return true;
}

__device__ bool intersectSphere(const scene::Ray &r, float rad, glm::vec3 pos, float& t)
{

	glm::vec3 op = pos - r.origin;
	float epsilon = 0.01f;
	float b = dot(op, r.dir);
	float disc = b*b - dot(op, op) + rad*rad;
	if (disc < 0)
		return 0;
	else
		//disc = sqrtf(disc);
		disc = __fsqrt_rn(disc);
	(t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);

	return t != 0;
}

__device__ inline bool
intersect(const scene::Ray& r,
	const struct scene::SceneData *const scene, glm::vec3& n, float& t, bool& light_emitter, glm::vec3& diff, glm::vec3& l)
{
  float inter_dist = 1000000.0;
  int inter_mat_idx = -1;

	glm::vec3 light_pos = glm::vec3(0.0f, -0.1f, 0.0f);
	glm::vec3 vertex[3];
  glm::vec3 normal[3];
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
        normal[v].x = scene->normals.data[3 * idx.normal_index];
        normal[v].y = scene->normals.data[3 * idx.normal_index + 1];
        normal[v].z = scene->normals.data[3 * idx.normal_index + 2];
			}
			if (intersectTriangle(vertex, r, t) && t < inter_dist)
			{
        inter_dist = t;
        n = normal[0];
        inter_mat_idx = mesh.material_ids.data[i / 3];
			}

			if (intersectSphere(r, 0.5f, light_pos, t))
			{
				light_emitter = true;
				l = light_pos;
				return true;
			}
		}

    // At least one intersection has been found.
    if (inter_mat_idx >= 0)
    {
      const scene::Material &const mat = scene->materials.data[inter_mat_idx];
      diff.x = mat.diffuse[0];
      diff.y = mat.diffuse[1];
      diff.z = mat.diffuse[2];
      return true;
    }

	}
	return false;
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
sample_lights(scene::Ray& r, glm::vec3 l, glm::vec3 color, glm::vec3 emission, float PDF, glm::vec3 normal)
{
	// Hardcoded for now, should be cleaned
	glm::vec3 light_pos = glm::vec3(0.0f, 0.2f, 0.0f);
	glm::vec3 L = glm::vec3(0.0f);
	int nb_lights = 1;
	for (int i = 0; i < nb_lights; i++)
	{
		// Hardcoded for now, should be cleaned
		if (l == light_pos)
			continue;

		float n_dot_l = glm::dot(normal, l);
		L += color * n_dot_l * emission / PDF;
	}

	return L;
}

__device__ inline glm::vec3 radiance(scene::Ray& r,
  const struct scene::SceneData *const scene, const scene::Camera * const cam,
  curandState* rand_state, int is_static, int static_samples)
{
  glm::vec3 acc = glm::vec3(0.0f, 0.0f, 0.0f);

  const int max_bounces = 1;// +is_static * (static_samples + 1);
  glm::vec3 col;
  for (int b = 0; b < max_bounces; b++)
  {
    glm::vec3 normal;
    glm::vec3 oriented_normal;
    glm::vec3 color = glm::vec3(0.2f, 0.2f, 0.1f);
    // Light energy emission
    glm::vec3 emission = glm::vec3(1.0f);
    // For energy compensation on Russian roulette
    glm::vec3 thoughput = glm::vec3(1.0f);
    glm::vec3 mat_reflectance = glm::vec3(1.0f);
    glm::vec3 l;
    float t = 100000;
    bool light_emitter = false;

    //float intersection = (float)intersect(r, scene, normal, t, light_emitter);
    if (intersect(r, scene, normal, t, light_emitter, col, l))
    {
      float cos_theta = glm::dot(normal, r.dir);
      glm::vec3 light_dir = glm::normalize(l - r.origin);
      float n_dot_l = glm::dot(normal, light_dir);
      oriented_normal = cos_theta < 0 ? normal : normal * -1.0f;

      //acc += mask * emission * (float)light_emitter * intersection;
      acc += emission * (float)light_emitter * thoughput;

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
      r.origin += r.dir * t;

      r.origin += oriented_normal * 0.03f;
      r.dir = d;

      //mask *= intersection * color + (1.0f - intersection) * 1.0f;
      //Lambert BRDF/PDF
      glm::vec3 BRDF = col * n_dot_l; // Divided by PI
      float PDF = cos_theta; // Divided by PI
                             //glm::vec3 BRDF = color;
      glm::vec3 direct_light = BRDF / PDF;
      thoughput *= direct_light;

      acc += thoughput * sample_lights(r, l, col, emission, PDF, normal);

      // Russian roulette
      float p = fmaxf(thoughput.x, fmaxf(thoughput.y, thoughput.z));
      if (r1 > p)
        return acc;

      thoughput *= 1.f / p;
    }
  }

  return col;
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

	//float screen_dist = __tanf(cam.fov_x * 0.5);

	//glm::vec3 look_at = glm::normalize(cam.dir - dir_offset);

	//glm::vec3 cx = glm::vec3(width * screen_dist / height, 0.0f, 0.0f);
	//glm::vec3 cy = glm::normalize(glm::cross(cx, look_at)) * screen_dist;

	glm::vec3 rad = glm::vec3(0.0f);
	scene::Ray r = generateRay(x, y, half_w, half_h, cam);
	/*r.dir = cx*((.25f + x) / width - .5f) + cy*((.25f + y) / height - .5f) + look_at;
	r.dir = glm::normalize(r.dir);
	r.origin = r.dir * 40.f + cam->position + offset / 10.f;*/

	int is_static = !moved;
	int static_samples = 1;
	int samples = 2 + is_static * static_samples;
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
