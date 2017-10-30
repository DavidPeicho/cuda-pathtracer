#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "scene.h"

// GPU
cudaError_t
raytrace(cudaArray_const_t array, const struct scene::SceneData *const scene, const scene::Camera * const cam,
  const unsigned int width, const unsigned int height, cudaStream_t stream,
  glm::vec3 *temporal_framebuffer, bool moved);
