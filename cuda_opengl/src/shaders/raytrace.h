#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "../scene/scene.h"
#include "cutils_math.h"

cudaError_t
raytrace(cudaArray_const_t array, const scene::Scenes *scenes, unsigned int scene_id,
  const scene::Cubemap& cubemap, const scene::Camera * const cam,
  const unsigned int width, const unsigned int height, cudaStream_t stream,
  float3 *temporal_framebuffer, bool moved);
