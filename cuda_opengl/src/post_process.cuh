#pragma once

#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <cuda.h>
#include <driver_types.h>

#include "driver/cuda_helper.h"

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
