#pragma once

#include "scene.h"

cudaError_t
raytrace(cudaArray_const_t array, const struct scene::SceneData *const scene,
  const unsigned int width, const unsigned int height, cudaStream_t stream, glm::vec3 offset, glm::vec3 dir_offset, glm::vec3 *temporal_framebuffer);