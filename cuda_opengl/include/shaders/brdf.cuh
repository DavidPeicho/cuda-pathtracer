#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "cutils_math.h"

#include "../driver/cuda_helper.h"

////////////////////////////////////////////////////////////////////////////////
// Implementations of Bidirectionnal Reflectance Distribution Functions (BRDFs)
////////////////////////////////////////////////////////////////////////////////

__device__ inline float3
brdf_lambert(float3 color)
{
  return color; // Divided by PI, but cancels out in the PDF
}

////////////////////////////////////////////////////////////////////////////////
// Implementations of Probability Distribution Functions (PDFs)
////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Computes the PDF lambert. This is only a constant.
/// </summary>
__device__ inline float
pdf_lambert()
{
  return 0.5f;
}

/// <summary>
/// Computes the Oren Nayar PDF. For now, it is not complete,
/// but we decided not to use it.
/// </summary>
__device__ inline float
pdf_oren_nayar()
{
  return 0.5f;
}
