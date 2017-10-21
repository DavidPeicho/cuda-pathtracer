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

__device__ inline bool
intersectTriangle(const glm::vec3 *vert, const scene::Ray &ray, glm::vec3& n, float& t)
{
	glm::vec3 v0v1 = vert[1] - vert[0];
	glm::vec3 v0v2 = vert[2] - vert[0];
	n = glm::normalize(glm::cross(v0v1, v0v2));
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

__device__ bool intersectSphere(const scene::Ray &r, float rad, glm::vec3 pos, float& t) {

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
	const struct scene::SceneData *const scene, glm::vec3& n, float& t, bool& light_emitter)
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
			if (intersectTriangle(vertex, r, n, t))
				return true;

			if (intersectSphere(r, 0.5f, glm::vec3(0.0f, 0.2f, 0.0f), t))
			{
				light_emitter = true;
				return true;
			}
		}
	}
	return false;
}

#define M_PI 3.14159265359f

__device__ inline glm::vec3 radiance(scene::Ray& r,
	const struct scene::SceneData *const scene, curandState* rand_state, int is_static, int static_samples)
{
	glm::vec3 mask = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec3 acc = glm::vec3(0.0f, 0.0f, 0.0f);

	const int max_bounces = 1 + is_static * (static_samples + 1);
	for (int b = 0; b < max_bounces; b++)
	{
		glm::vec3 normal;
		glm::vec3 oriented_normal;
		glm::vec3 color = glm::vec3(0.2f, 0.2f, 0.1f);
		// Light energy emission
		glm::vec3 emission = glm::vec3(2.0f);
		// For energy compensation on Russian roulette
		glm::vec3 thoughput = glm::vec3(1.0f);
		glm::vec3 mat_reflectance = glm::vec3(1.0f);
		float t = 100000;
		bool light_emitter = false;

		//float intersection = (float)intersect(r, scene, normal, t, light_emitter);
		if (intersect(r, scene, normal, t, light_emitter))
		{
			float cos_theta = glm::dot(normal, r.dir);
			oriented_normal = cos_theta < 0 ? normal : normal * -1.0f;

			//acc += mask * emission * (float)light_emitter * intersection;
			acc += mask * emission * (float)light_emitter * thoughput;

			float r2 = sqrtf(curand_uniform(rand_state));
			float r1 = 2.0f * M_PI * curand_uniform(rand_state);

			// Russian roulette
			float p = fmaxf(thoughput.x, fmaxf(thoughput.y, thoughput.z));
			if (r2 > p)
				return acc;

			thoughput *= 1 / p;

			float r2_squared = sqrtf(r2);

			glm::vec3 u = glm::normalize(glm::cross(fabs(oriented_normal.x) > .1 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f), oriented_normal));
			glm::vec3 v = glm::cross(oriented_normal, u);

			glm::vec3 d = glm::normalize(u * cos(r1) * r2_squared + v * sin(r1) * r2_squared + oriented_normal * sqrtf(1 - r2));

			r.origin += r.dir * t;

			r.origin += oriented_normal * 0.03f;
			r.dir = d;

			//mask *= intersection * color + (1.0f - intersection) * 1.0f;
			glm::vec3 BRDF = 2.0f * mat_reflectance * cos_theta * color;
			mask *= BRDF;
		}
	}

	return acc;
}

__global__ void
kernel(const unsigned int width, const unsigned int height,
	const unsigned int half_w, const unsigned int half_h,
	const scene::SceneData *const scene, unsigned int hash_seed,
	glm::vec3 offset, glm::vec3 dir_offset, int frame_nb, glm::vec3 *temporal_framebuffer, bool moved)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const unsigned int tid = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	union rgba_24 rgbx;
	rgbx.a = 0;

	curandState rand_state;
	curand_init(hash_seed + tid, 0, 0, &rand_state);

	struct scene::Camera *cam = scene->cam;
	float screen_dist = half_w / __tanf(cam->fov_x * 0.5);

	glm::vec3 look_at = glm::normalize(cam->dir + dir_offset);

	glm::vec3 cx = glm::vec3(width * cam->fov_x / height, 0.0f, 0.0f);
	glm::vec3 cy = glm::normalize(glm::cross(cx, look_at)) * cam->fov_x;

	glm::vec3 rad = glm::vec3(0.0f);
	scene::Ray r;
	r.dir = cx*((.25f + x) / width - .5f) + cy*((.25f + y) / height - .5f) + look_at;
	r.dir = glm::normalize(r.dir);
	r.origin = r.dir * 40.f + cam->position + offset / 10.f;

	int is_static = !moved;
	int static_samples = 1;
	int samples = 2 + is_static * static_samples;
	for (int i = 0; i < samples; i++)
		rad += radiance(r, scene, &rand_state, is_static, static_samples);

	rad /= samples;

	rad = glm::clamp(rad, 0.0f, 1.0f);

	int i = (height - y - 1)*width + x;
	temporal_framebuffer[i] *= is_static;
	temporal_framebuffer[i] += rad;

	rad = temporal_framebuffer[i] / (float)frame_nb;

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
raytrace(cudaArray_const_t array, const scene::SceneData *const scene,
	const unsigned int width, const unsigned int height, cudaStream_t stream,
	glm::vec3 offset, glm::vec3 dir_offset, glm::vec3 *temporal_framebuffer, bool moved)
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
		kernel << <nb_blocks, threads_per_block, 0, stream >> > (width, height,
			width / 2, height / 2, scene, WangHash(seed), offset, dir_offset, seed, temporal_framebuffer, moved);

	return cudaSuccess;
}