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

HOST_DEVICE inline bool
intersectTriangle(const glm::vec3 *vert, const scene::Ray &ray, glm::vec3& n, float& t)
{
	glm::vec3 v0v1 = vert[1] - vert[0];
	glm::vec3 v0v2 = vert[2] - vert[0];
	n = glm::normalize(glm::cross(v0v1, v0v2));
	glm::vec3 p_vec = glm::cross(ray.dir, v0v2);
	double det = glm::dot(v0v1, p_vec);
	if (det > -0.0000001 && det < 0.0000001)
		return false;

	double inv_det = 1 / det;
	glm::vec3 t_vec = ray.origin - vert[0];
	double u = glm::dot(t_vec, p_vec) * inv_det;
	if (u < 0 || u > 1) return false;

	glm::vec3 qvec = glm::cross(t_vec, v0v1);
	double v = glm::dot(ray.dir, qvec) * inv_det;
	if (v < 0 || u + v > 1) return false;

	t = glm::dot(v0v2, qvec) * inv_det;
	return true;
}

HOST_DEVICE inline bool
intersect(const scene::Ray& r,
	const struct scene::SceneData *const scene, glm::vec3& n, float& t)
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
		}
	}
	return false;
}

#define M_PI 3.14159265359f

__device__ inline glm::vec3 radiance(scene::Ray& r,
	const struct scene::SceneData *const scene, curandState* rand_state)
{
	glm::vec3 mask = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec3 acc = glm::vec3(0.0f, 0.0f, 0.0f);

	const int max_bounces = 1;
	for (int b = 0; b < max_bounces; b++)
	{
		glm::vec3 normal;
		glm::vec3 oriented_normal;
		glm::vec3 new_dir;
		glm::vec3 inter_point;
		glm::vec3 color = glm::vec3(0.9f, 0.2f, 0.1f);
		glm::vec3 emission = color;
		float t = 100000;

		if (intersect(r, scene, normal, t))
		{
			oriented_normal = glm::dot(normal, r.dir) < 0 ? normal : normal * -1.0f;

			acc += mask * emission;

			float r1 = 2.0f * M_PI * curand_uniform(rand_state);
			float r2 = curand_uniform(rand_state);

			float r2_squared = sqrtf(r2);

			glm::vec3 u = glm::normalize(glm::cross(fabs(oriented_normal.x) > .1 ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f), oriented_normal));
			glm::vec3 v = glm::cross(oriented_normal, u);

			glm::vec3 d = glm::normalize(u * cos(r1) * r2_squared + v * sin(r1) * r2_squared + oriented_normal * sqrtf(1 - r2));

			new_dir = r.origin + r.dir * t;

			r.origin += oriented_normal * 0.03f;
			r.dir = d;

			mask *= color;
		}
	}

	return acc;
}

__global__ void
kernel(const unsigned int width, const unsigned int height,
	const unsigned int half_w, const unsigned int half_h,
	const scene::SceneData *const scene, unsigned int hash_seed)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const unsigned int tid = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	union rgba_24 rgbx;
	rgbx.a = 0;

	curandState rand_state;
	curand_init(hash_seed + tid, 0, 0, &rand_state);

	struct scene::Camera *cam = scene->cam;
	float screen_dist = half_w / __tanf(cam->fov_x);

	scene::Ray r;
	r.origin = scene->cam->position;
	r.dir.z = screen_dist;
	r.dir.x = x - half_w;
	r.dir.y = y - half_h;
	r.dir = glm::normalize(r.dir);

	glm::vec3 rad = radiance(r, scene, &rand_state);

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
	const unsigned int width, const unsigned int height, cudaStream_t stream)
{
	static unsigned int seed = 0;
	seed++;

	cudaBindSurfaceToArray(surf, array);

	// Register occupancy : nb_threads = regs_per_block / 32
	// Shared memory occupancy : nb_threads = shared_mem / 32
	// Block size occupancy 

	// TODO: We should get into account GPU info, such as number of registers,
	// shared memory size, warp size, etc...
	dim3 threads_per_block(8, 8);
	dim3 nb_blocks(width / 8, height / 8);

	if (nb_blocks.x > 0 && nb_blocks.y > 0)
		kernel << <nb_blocks, threads_per_block, 0, stream >> > (width, height,
			width / 2, height / 2, scene, WangHash(seed));

	return cudaSuccess;
}