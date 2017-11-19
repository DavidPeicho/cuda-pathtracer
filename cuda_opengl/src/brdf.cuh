#pragma once

#define GLM_FORCE_CUDA
#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <cuda.h>
#include <driver_types.h>

#include "driver/cuda_helper.h"

// BRDFs
// Diffuse
__device__ inline glm::vec3
brdf_oren_nayar(float n_dot_v, float n_dot_l, glm::vec3 light_dir, glm::vec3 view_dir,
						  glm::vec3 n, float roughness, float metalness, glm::vec3 base_color)
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

	glm::vec3 color = base_color;
	color = glm::mix(color, glm::vec3(0.0), metalness);

	return glm::vec3(glm::vec3(color) * glm::vec3(L1));
}

__device__ inline glm::vec3 brdf_lambert(glm::vec3 color)
{
	return color; // Divided by PI, but cancels out in the PDF
}

// PDFs
__device__ inline float pdf_lambert()
{
	return 0.5f;
}

__device__ inline float pdf_oren_nayar()
{
	return 0.5f; // For now
}
