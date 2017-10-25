#pragma once

#include <cuda.h>
#include <driver_types.h>

#include "scene.h"

// GPU
cudaError_t
raytrace(cudaArray_const_t array, const struct scene::SceneData *const scene,
  const unsigned int width, const unsigned int height, cudaStream_t stream);


// CPU
void
raytrace_cpu(unsigned char *pixels, const struct scene::SceneData *const scene,
  const unsigned int width, const unsigned int height);
