#include <cuda_runtime.h>

#include <math_functions.h>

#include <curand.h>
#include <curand_kernel.h>

#include <stdbool.h>

#include "../utils.h"
#include "../scene/scene_data.h"

#include <iostream>

#include "brdf.cuh"
#include "post_process.cuh"

surface<void, cudaSurfaceType2D> surf;
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

struct __align__(8) IntersectionData
{
  float3 normal;
  float3 surface_normal;
  float3 tangent;
  float3 diffuse_col;
  float2 uv;
  const scene::LightProp *light;
  float dist;
  float specular_col;
  float ior;
};

/*
  The sampleTexture() method uses overloading to sample
  several type of texture easily. It allows us to sample
  3-channels texture as well as 1-channel textures.
*/

__device__ inline int
getTextureIdx(const scene::Texture &texture, const float2& uv)
{
  int x = uv.x * (texture.w - 1);
  int y = uv.y * (texture.h - 1);
  return (y * texture.w + x) * texture.nb_chan;
}

__device__ inline void
sampleTexture(const scene::Buffer<scene::Texture>& textures,
  int tex_id, const float2& uv, float3& out)
{
  const scene::Texture &tex = textures.data[tex_id];
  int idx = getTextureIdx(tex, uv);

  out.x = tex.data[idx];
  out.y = tex.data[idx + 1];
  out.z = tex.data[idx + 2];
}

__device__ inline void
sampleTexture(const scene::Buffer<scene::Texture>& textures,
  int tex_id, const float2& uv, float4& out)
{
  const scene::Texture &tex = textures.data[tex_id];
  int idx = getTextureIdx(tex, uv);

  out.x = tex.data[idx];
  out.y = tex.data[idx + 1];
  out.z = tex.data[idx + 2];
  out.w = tex.data[idx + 3];
}

HOST_DEVICE inline scene::Ray
generateRay(const int x, const int y,
            const int half_w, const int half_h, scene::Camera &cam)
{
  float screen_dist = half_w / std::tanf(cam.fov_x * 0.5f);

  scene::Ray ray;
  ray.origin = cam.position;

  cam.u = normalize(cross(cam.dir, make_float3(0.0, -1.0, 0.0)));
  cam.v = normalize(cross(cam.u, cam.dir));

  cam.u *= -1.0;

  float3 screen_pos = cam.position + (cam.dir * screen_dist)
    + (cam.u * (float)(x - half_w)) + (cam.v * (float)(y - half_h));

  ray.dir = screen_pos - cam.position;
  ray.dir = normalize(ray.dir);

  return ray;
}

__device__ inline bool
intersectTriangle(const scene::Face &face, float3 &out_normal,
  float2 &out_uv, const scene::Ray &ray, float& t)
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
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(t_vec, v0v1);
	float v = dot(ray.dir, qvec) * inv_det;
	if (v < 0 || u + v > 1) return false;

  // Interpolates normals
  out_normal = (1.0f - u - v) * face.normals[0] + u * face.normals[1] + v * face.normals[2];
  //out_normal = face.normals[0];
  // Interpolates uvs
  out_uv = (1.0f - u - v) * face.texcoords[0] + u * face.texcoords[1] + v * face.texcoords[2];
  out_uv = mod(out_uv, 1.0);

	t = dot(v0v2, qvec) * inv_det;
	return true;
}

__device__ bool
intersectSphere(const scene::Ray &r, const scene::LightProp & light, float& t)
{
    static const float epsilon = 0.01f;

	float3 op = light.vec - r.origin;
	float b = dot(op, r.dir);
	float disc = b*b - dot(op, op) + light.radius * light.radius;
	if (disc < 0.0) return 0;

	disc = __fsqrt_rn(disc);
	(t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);

	return t != 0.0;
}

__device__ inline bool
intersect(const scene::Ray& r, const scene::Scenes *scenes, unsigned int scene_id,
  IntersectionData &intersection)
{
  static const float MAX_DIST = 100000.0;

  const scene::SceneData *scene = scenes->scenes[scene_id];
  const scene::Buffer<scene::Texture>& textures = scenes->textures;

  float inter_dist = MAX_DIST;
  intersection.dist = MAX_DIST;

  float3 normal;
  float2 uv;

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
  for (unsigned l = 0; l < scene->lights.size; ++l)
  {
    // Checks lights intersection
    const scene::LightProp & light = scene->lights.data[l];
    if (intersectSphere(r, light, inter_dist)
        && inter_dist < intersection.dist && inter_dist >= 0.0)
    {
      intersection.light = &light;
      intersection.dist = inter_dist;
      intersection.diffuse_col = make_float3(light.color.x, light.color.y, light.color.z);
      intersection.normal = normalize(light.vec - (inter_dist * r.dir));
      intersection.surface_normal = intersection.surface_normal;
      inter_mat = nullptr;
    }
  }

  // Samples texture & computes normal mapping only once,
  // when we are sure this is the closest intersection.
  if (inter_mat)
  {
    // Fetches diffuse color from texture
    float4 fetch;
    sampleTexture(textures, inter_mat->diffuse_spec_map, intersection.uv, fetch);
    intersection.diffuse_col.x = fetch.x;
    intersection.diffuse_col.y = fetch.y;
    intersection.diffuse_col.z = fetch.z;

    intersection.specular_col = fetch.w;

    // Computes normal perturbated by normal map
    if (inter_mat->normal_map >= 0)
    {
      float3 normal;
      sampleTexture(textures, inter_mat->normal_map, intersection.uv, normal);
      intersection.normal = normalize((normal * 2.0f) - 1.0f);

      float3 binormal = normalize(cross(intersection.tangent, intersection.surface_normal));

      mat3 tbn;
      tbn.x = intersection.tangent;
      tbn.y = -binormal;
      tbn.z = intersection.surface_normal;

      //intersection.normal = normalize((normal * 2.0f) - 1.0f);
      intersection.normal = tbn * intersection.normal;
    }
  }

	return intersection.dist < MAX_DIST;
}

#ifndef M_PI
#define M_PI 3.14159265359f
#endif
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

__device__ inline float3 radiance(scene::Ray& r,
  const struct scene::Scenes *scenes, unsigned int scene_id,
  const scene::Camera * const cam, curandState* rand_state,
  int is_static, int static_samples)
{
  float3 acc = make_float3(0.0f);
  // For energy compensation on Russian roulette
  float3 throughput = make_float3(1.0f);

  // Contains information about each intersection.
  // This will be updated at each call to 'intersect'.
  IntersectionData inter;

  // Max bounces
  // Bounce more when the camera is not moving
  const int max_bounces = 1 + is_static * (static_samples + 1);
  for (int b = 0; b < max_bounces; b++)
  {
	  float3 oriented_normal;

	  float r1 = curand_uniform(rand_state);
    if (intersect(r, scenes, scene_id, inter))
	  {
		  float cos_theta = dot(inter.normal, r.dir);
		  oriented_normal = cos_theta < 0 ? inter.normal : inter.normal * -1.0f;

		  float3 up = make_float3(0.0, 1.0, 0.0);
		  float3 right = cross(up, inter.normal);
		  up = cross(inter.normal, right);

		  // Oren-Nayar diffuse
		  //BRDF = brdf_oren_nayar(cos_theta, cos_theta, light_dir, r.dir, oriented_normal, 0.5f, 0.5f, inter.diffuse_col);

		  // Specular ray
		  // Computed everytime and then used to simulate roughness by concentrating rays towards it
		  float3 spec = reflect(r.dir, inter.normal);
		  float PDF = pdf_lambert(); // Divided by PI
		  //Lambert BRDF/PDF
		  float3 BRDF = brdf_lambert(inter.diffuse_col); // Divided by PI
		  float3 direct_light = BRDF / PDF;
		  // Default IOR (Index Of Refraction) is 1.0f
		  if (inter.ior == 1.0f || inter.light != NULL)
		  {
			  // Accumulate light emission
			  if (inter.light != NULL)
			  {
				  BRDF = make_float3(inter.light->color.x, inter.light->color.y, inter.light->color.z);

				  acc += BRDF * inter.light->emission * throughput;
			  }

			  // Sample the hemisphere with a random ray
			  float phi = 2.0f * M_PI * curand_uniform(rand_state);//glm::mix(1.0f - inter.specular_col, inter.specular_col, 0.1f);

			  float sin_t = __fsqrt_rn(r1);
			  float cos_t = __fsqrt_rn(1.f - r1);

			  // u, v and oriented_normal form the base of the hemisphere
			  float3 u = normalize(cross(fabs(oriented_normal.x) > .1 ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f), oriented_normal));
			  float3 v = cross(oriented_normal, u);

			  //Diffuse hemishphere reflection
			  float3 d = normalize(v * sin_t * __cosf(phi) + u * __sinf(phi) * sin_t + oriented_normal * cos_t);

			  r.origin += r.dir * inter.dist;

			  // Mix the specular and random diffuse ray by the "specular_col" amount to approximate roughness
			  r.dir = mix(d, spec, inter.specular_col);

			  // Avoids self intersection
			  r.origin += r.dir * 0.03f;

			  throughput *= direct_light;
		  }
		  else
		  {
			  // Transmision
			  // n1: IOR of exterior medium
			  float n1 = 1.0f; // sin theta2
			  // n1: IOR of entering medium
			  float n2 = inter.ior; // sin theta1
			  float c1 = dot(oriented_normal, r.dir);
			  bool entering = dot(inter.normal, oriented_normal) > 0;
			  // Snell's Law
			  float eta = entering ? n1 / n2 : n2 / n1;
			  float eta_2 = eta * eta;

			  float c2_term = 1.0f - eta_2 * (1.0f - c1 * c1);
			  // Total Internal Reflection
			  if (c2_term < 0.0f)
			  {
				  r.origin += oriented_normal * inter.dist / 100.f;

				  //return glm::vec3(0.0f, 1.0f, 0.0f);
				  r.dir = spec;
			  }
			  else
			  {
				  // Schlick R0
				  float R0 = (n2 - n1) / (n1 + n2);
				  R0 *= R0;
				  float c2 = __fsqrt_rn(c2_term);
				  float3 T = normalize(eta * r.dir + (eta * c1 - c2) * oriented_normal);

				  float f_cos_theta = 1.0f - (entering ? -c1 : dot(T, inter.normal));
				  f_cos_theta = powf(cos_theta, 5.0f);
				  // Fresnel-Schlick approximation for the reflection amount
				  float f_r = R0 + (1.0f - R0) * f_cos_theta;

				  // If reflection
				  // Not exactly sure why "0.25f" works better than "f_r"...
				  if (curand_uniform(rand_state) < 0.25f)
				  {
					  throughput *= f_r * direct_light;

					  r.origin += oriented_normal * inter.dist / 100.f;

					  r.dir = spec;// mix(d, spec, inter.specular_col);
				  }
				  else // Transmission
				  {
					  // Energy conservation
					  float f_t = 1.0f - f_r;

					  throughput *= f_t * direct_light;

					  // We're inside a mesh doing transmission, so we try to reduce the bias as much as possible
					  // or the ray could get outside of the mesh which makes no sense
					  r.origin += oriented_normal * inter.dist / 10000.f;

					  r.dir = T;
				  }
			  }
		  }
	  }
	  else
	  {
		  // Accumulate Environment map's contribution (approximated as many far away lights)
		  auto val = texCubemap(cubemap_ref, r.dir.x, r.dir.y, -r.dir.z);
		  acc += make_float3(val.x, val.y, val.z) * throughput;
	  }

	  // Russian roulette for early path termination
	  float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
	  if (r1 > p && b > 1)
		  return acc;

	  throughput *= __fdividef(1.0f, p);
  }

  return acc;
}

__global__ void
kernel(const unsigned int width, const unsigned int height,
	const scene::Scenes *scenes, unsigned int scene_id,
  scene::Camera cam, unsigned int hash_seed, int frame_nb,
  float3 *temporal_framebuffer, bool moved)
{
	const unsigned int half_w = width / 2;
  	const unsigned int half_h = height / 2;

  	const int x = blockDim.x * blockIdx.x + threadIdx.x;
  	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const unsigned int tid = (blockIdx.x + blockIdx.y * gridDim.x)
    * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	union rgba_24 rgbx;
	rgbx.a = 0.0;

	curandState rand_state;
	curand_init(hash_seed + tid, 0, 0, &rand_state);

	scene::Ray r = generateRay(x, y, half_w, half_h, cam);

	// Depth-Of-Field
	camera_dof(r, cam, &rand_state);

	int is_static = !moved;
	int static_samples = 1;

  float3 rad = radiance(r, scenes, scene_id, &cam, &rand_state, is_static, static_samples);
/*=======
	int samples = 2 + is_static * static_samples;

  float3 rad = make_float3(0.0f);
	for (int i = 0; i < samples; i++)
	  rad = radiance(r, scenes, scene_id, &cam, &rand_state, is_static, static_samples);

  rad /= samples;
>>>>>>> feature/gui*/
	rad = clamp(rad, 0.0f, 1.0f);

	// Accumulation buffer for when the camera is static
	// This makes the image converge
	int i = (height - y - 1) * width + x;

  // Zero-out if the camera is moving to reset the buffer
	temporal_framebuffer[i] *= is_static;
	temporal_framebuffer[i] += rad;

	rad = temporal_framebuffer[i] / (float)frame_nb;

	// Tone Mapping + White Balance
	rad = exposure(rad);
	// Gamma Correction
	rad = pow(rad, 1.0f / 2.2f);

    rgbx.r = rad.x * 255;
    rgbx.g = rad.y * 255;
    rgbx.b = rad.z * 255;

	surf2Dwrite(rgbx.b32,
		surf,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero);
}

// Very nice and fast PRNG
// Credit: Thomas Wang
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
raytrace(cudaArray_const_t array, const scene::Scenes *scenes, unsigned int scene_id,
  const std::vector<scene::Cubemap>& cubemaps, int cubemap_id,
  const scene::Camera * const cam, const unsigned int width, const unsigned int height,
  cudaStream_t stream, float3 *temporal_framebuffer, bool moved)
{
	// Seed for the Wang Hash
	static unsigned int seed = 0;

	if (moved) seed = 0;
	seed++;

	cudaBindSurfaceToArray(surf, array);

  const scene::Cubemap& cubemap = cubemaps[cubemap_id];
  cubemap_ref.addressMode[0] = cudaAddressModeWrap;
  cubemap_ref.addressMode[1] = cudaAddressModeWrap;
  cubemap_ref.filterMode = cudaFilterModeLinear;
  cubemap_ref.normalized = true;
  cudaBindTextureToArray(cubemap_ref, cubemap.cubemap, cubemap.cubemap_desc);

	// Register occupancy : nb_threads = regs_per_block / 32
	// Shared memory occupancy : nb_threads = shared_mem / 32
	// Block size occupancy

	// TODO: We should get into account GPU info, such as number of registers,
	// shared memory size, warp size, etc...
	dim3 threads_per_block(16, 16);
	dim3 nb_blocks(width / threads_per_block.x + 1, height / threads_per_block.y + 1);

  if (nb_blocks.x > 0 && nb_blocks.y > 0)
    kernel << <nb_blocks, threads_per_block, 0, stream >> > (width, height, scenes, scene_id , *cam,
      WangHash(seed), seed, temporal_framebuffer, moved);

	return cudaSuccess;
}
