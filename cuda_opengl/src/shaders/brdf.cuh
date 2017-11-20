#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "cutils_math.h"

#include "../driver/cuda_helper.h"

// BRDFs
// Diffuse
__device__ inline float3
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
}

//__device__ inline float3 btdf_refraction()
__device__ inline float3
brdf_lambert(float3 color)
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
