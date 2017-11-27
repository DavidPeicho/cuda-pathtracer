#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "cutils_math.h"

#include "../driver/cuda_helper.h"

////////////////////////////////////////////////////////////////////////////////
// Implementations of Bidirectionnal Reflectance Distribution Functions (BRDFs)
////////////////////////////////////////////////////////////////////////////////

// Diffuse
// Very nice and Physicaly-Based BRDF, but uses too much instructions
/*__device__ inline float3
brdf_oren_nayar(float n_dot_v, float n_dot_l, float3 light_dir, float3 view_dir,
						    float3 n, float roughness, float metalness, float3 base_color)
{
	float angle_v_n = acos(n_dot_v);
	float angle_l_n = acos(n_dot_l);

	float alpha = fmax(angle_v_n, angle_l_n);
	float beta = fmin(angle_v_n, angle_l_n);
	float gamma = dot(view_dir - n * n_dot_v, light_dir - n * n_dot_l);

	float roughness_2 = roughness * roughness;

	float A = 1.0 - 0.5 * (roughness_2 / (roughness_2 + 0.57));
	float B = 0.45 * (roughness_2 / (roughness_2 + 0.09));
	float C = sin(alpha) * tan(beta);

	float L1 = fmax(0.0, n_dot_l) * (A + B * fmax(0.0, gamma) * C);

	float3 color = base_color;
	color = mix(color, make_float3(0.0), metalness);

	return float3(float3(color) * make_float3(L1));
}*/

__device__ inline float3
brdf_lambert(float3 color)
{
	return color; // Divided by PI, but cancels out in the PDF
}

////////////////////////////////////////////////////////////////////////////////
// Implementations of Probability Distribution Functions (PDFs)
////////////////////////////////////////////////////////////////////////////////

__device__ inline float pdf_lambert()
{
	return 0.5f;
}

__device__ inline float pdf_oren_nayar()
{
	return 0.5f; // For now
}

// A trick to converge faster, but at the cost of more intersections
// Would work better in an offline Path-tracer
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