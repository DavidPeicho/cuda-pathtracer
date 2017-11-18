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

#include "brdf.cuh"
#include "post_process.cuh"

surface<void, cudaSurfaceType2D> surf;
//texture<float, cudaTextureTypeCubemap> cubemap_ref;
//texture<uint4, cudaTextureTypeCubemap> cubemap_ref;
texture<float4, cudaTextureTypeCubemap> cubemap_ref;

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
  glm::vec3 surface_normal;
  glm::vec3 tangent;
  glm::vec3 diffuse_col;
  glm::vec2 uv;
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

__device__ inline void
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
intersectTriangle(const scene::Face &face, glm::vec3 &out_normal,
  glm::vec2 &out_uv, const scene::Ray &ray, float& t)
{
	glm::vec3 v0v1 = face.vertices[1] - face.vertices[0];
	glm::vec3 v0v2 = face.vertices[2] - face.vertices[0];
	glm::vec3 p_vec = glm::cross(ray.dir, v0v2);
	float det = glm::dot(v0v1, p_vec);
	if (det < 0.0000001)
		return false;

	float inv_det = __fdividef(1.f, det);
	glm::vec3 t_vec = ray.origin - face.vertices[0];
	float u = glm::dot(t_vec, p_vec) * inv_det;
	if (u < 0 || u > 1) return false;

	glm::vec3 qvec = glm::cross(t_vec, v0v1);
	float v = glm::dot(ray.dir, qvec) * inv_det;
	if (v < 0 || u + v > 1) return false;

  // Interpolates normals
  //out_normal = (1.0f - u - v) * v_norm[0] + u * v_norm[1] + v * v_norm[2];
  out_normal = face.normals[0];
  // Interpolates uvs
  out_uv = (1.0f - u - v) * face.texcoords[0] + u * face.texcoords[1] + v * face.texcoords[2];
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

  glm::vec3 normal;
  glm::vec2 uv;

  const scene::Material *inter_mat = nullptr;

  // Checks meshes intersection
	for (size_t m = 0; m < scene->meshes.size; ++m)
	{
		const auto &mesh = scene->meshes.data[m];
    for (size_t i = 0; i < mesh.faces.size; ++i)
    {
      const auto &face = mesh.faces.data[i];
      if (intersectTriangle(face, normal, uv, r, inter_dist)
        && inter_dist < intersection.dist && inter_dist > 0.0)
      {
        inter_mat = &scene->materials.data[face.material_id];
        intersection.normal = normal;
        intersection.surface_normal = normal;
        intersection.tangent = face.tangent;

        intersection.uv = uv;
        intersection.dist = inter_dist;
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
      intersection.surface_normal = intersection.normal;
      inter_mat = nullptr;
    }
  }

  // Samples texture & computes normal mapping only once,
  // when we are sure this is the closest intersection.
  if (inter_mat)
  {
    // Fetches diffuse color from texture
    sampleTexture(scene->textures, inter_mat->diffuse_spec_map, intersection.uv, intersection.diffuse_col);

    // Computes normal perturbated by normal map
    if (inter_mat->normal_map >= 0)
    {
      glm::vec3 normal;
      sampleTexture(scene->textures, inter_mat->normal_map, intersection.uv, normal);
      intersection.normal = glm::normalize(normal * 2.0f - 1.0f);

      glm::vec3 binormal = glm::normalize(glm::cross(intersection.tangent, intersection.surface_normal));

      glm::mat3 tbn;
      tbn[0] = intersection.tangent;
      tbn[1] = -binormal;
      tbn[2] = intersection.surface_normal;

      intersection.normal = tbn * intersection.normal;
    }
  }

	return intersection.dist < MAX_DIST;
}

#define M_PI 3.14159265359f

/*__device__ inline glm::vec3
sample_lights(scene::Ray& r, const struct scene::SceneData *const scene,
			  float PDF, const IntersectionData& inter)
{
	glm::vec3 L = glm::vec3(0.0f);
	const scene::Buffer<scene::LightProp>& lights = scene->lights;
	for (int i = 0; i < lights.size; i++)
	{
		const scene::LightProp& const l = lights.data[i];
		scene::LightProp light;
		glm::vec3 light_dir = l.vec - r.origin;

		IntersectionData in;

		scene::Ray r_l;
		r_l.dir = light_dir;
		r_l.origin = r.origin;
		if (intersect(r_l, scene, in))
		{
			if (!inter.is_light)
				continue;

			if (l.vec == inter.light->vec)
				continue;


			float n_dot_l = glm::dot(in.normal, light_dir);
			L += inter.diffuse_col * n_dot_l * l.emission / PDF;
		}
	}

	return L;
}*/

__device__ inline glm::vec3 radiance(scene::Ray& r,
  const struct scene::SceneData *const scene, const scene::Camera * const cam,
  curandState* rand_state, int is_static, int static_samples)
{
  glm::vec3 acc = glm::vec3(0.0f);
  // For energy compensation on Russian roulette
  glm::vec3 thoughput = glm::vec3(1.0f);

  // Contains information about each intersection.
  // This will be updated at each call to 'intersect'.
  IntersectionData inter;

  //const int max_bounces = 1 + is_static * (static_samples + 1);
  const int max_bounces = 1;
  for (int b = 0; b < max_bounces; b++)
  {
	  glm::vec3 oriented_normal;

	  if (intersect(r, scene, inter))
	  {
		  float cos_theta = glm::dot(inter.normal, r.dir);
		  oriented_normal = cos_theta < 0 ? inter.normal : inter.normal * -1.0f;
		  oriented_normal = inter.normal;

		  //acc += mask * emission * (float)light_emitter * intersection;
		  float r1 = curand_uniform(rand_state);

		  glm::vec3 BRDF;

		  glm::vec3 up = glm::vec3(0.0, 1.0, 0.0);
		  glm::vec3 right = glm::cross(up, inter.normal);
		  up = glm::cross(inter.normal, right);

		  // Oren-Nayar diffuse
		  //BRDF = brdf_oren_nayar(cos_theta, cos_theta, light_dir, r.dir, oriented_normal, 0.5f, 0.5f, inter.diffuse_col);

		  if (inter.is_light)
		  {
			  acc += inter.light->emission * thoughput;

			  float p = fmaxf(thoughput.x, fmaxf(thoughput.y, thoughput.z));
			  if (r1 > p && b > 1)
				  return acc;

			  thoughput *= 1.0 / p;

			  BRDF = glm::vec3(inter.light->color[0], inter.light->color[1], inter.light->color[2]);
		  }
		  else
			  BRDF = brdf_lambert(inter.diffuse_col); // Divided by PI

		  glm::vec3 d;
		  float phi = 2.0f * M_PI * curand_uniform(rand_state) * glm::mix(1.0f - inter.specular_col, inter.specular_col, 0.1f);

		  float sin_t = sqrtf(r1);
		  float cos_t = sqrt(1.f - r1);

		  glm::vec3 u = glm::normalize(glm::cross(fabs(oriented_normal.x) > .1 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f), oriented_normal));
		  glm::vec3 v = glm::cross(oriented_normal, u);

		  //Diffuse hemishphere reflection
		  d = glm::normalize(v * sin_t * cos(phi) + u * sin(phi) * sin_t + oriented_normal * cos_t);

		  r.origin += r.dir * inter.dist;

		  r.origin += oriented_normal * 0.03f;
		  r.dir = d;

		  //Lambert BRDF/PDF
		  float PDF = pdf_lambert(); // Divided by PI
		  glm::vec3 direct_light = BRDF / PDF;

		  thoughput *= direct_light;

		  //acc = glm::vec3(glm::mix(inter.diffuse_col[0], inter.specular_col, 0.4));
		  //acc += thoughput;// *sample_lights(r, scene, PDF, inter);

		  /*if (is_static)
			acc += thoughput * sample_lights(r, scene, PDF, inter);*/

		  // Russian roulette
		  float p = fmaxf(thoughput.x, fmaxf(thoughput.y, thoughput.z));
		  if (r1 > p && b > 1)
			  return acc;

		  thoughput *= 1.0 / p;
	  }
    else {
      auto val = texCubemap(cubemap_ref, r.dir.x, r.dir.y, r.dir.z);
      acc += glm::vec3(val.x, val.y, val.z);
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

	//Focus dist
	/*glm::vec3 focalPoint = 2.f * r.dir;
	float randomAngle = curand_uniform(&rand_state) * 2.0f * M_PI;
	float randomRadius = curand_uniform(&rand_state) * 0.5f;
	glm::vec3  randomAperturePos = ( cos(randomAngle) * cam.u + sin(randomAngle) * cam.v ) * randomRadius;
	// point on aperture to focal point
	glm::vec3 finalRayDir = normalize(focalPoint - randomAperturePos);

	r.origin += randomAperturePos;
	r.dir = finalRayDir;*/

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
raytrace(cudaArray_const_t array, const scene::SceneData *const cpu_scene,
  const scene::SceneData *const gpu_scene, const scene::Camera * const cam,
	const unsigned int width, const unsigned int height, cudaStream_t stream,
	glm::vec3 *temporal_framebuffer, bool moved)
{
	static unsigned int seed = 0;

	if (moved)
		seed = 0;

	seed++;

	cudaBindSurfaceToArray(surf, array);

  cubemap_ref.addressMode[0] = cudaAddressModeWrap;
  cubemap_ref.addressMode[1] = cudaAddressModeWrap;
  cubemap_ref.filterMode = cudaFilterModeLinear;
  cubemap_ref.normalized = true;
  cudaBindTextureToArray(cubemap_ref, cpu_scene->cubemap, cpu_scene->cubemap_desc);

	// Register occupancy : nb_threads = regs_per_block / 32
	// Shared memory occupancy : nb_threads = shared_mem / 32
	// Block size occupancy

	// TODO: We should get into account GPU info, such as number of registers,
	// shared memory size, warp size, etc...
	dim3 threads_per_block(16, 16);
	dim3 nb_blocks(width / threads_per_block.x, height / threads_per_block.y);

	if (nb_blocks.x > 0 && nb_blocks.y > 0)
		kernel << <nb_blocks, threads_per_block, 0, stream >> > (width, height, gpu_scene, *cam,
      WangHash(seed), seed, temporal_framebuffer, moved);

	return cudaSuccess;
}
