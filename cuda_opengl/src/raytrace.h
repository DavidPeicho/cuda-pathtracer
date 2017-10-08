#pragma once

cudaError_t
raytrace(cudaArray_const_t array, const unsigned int width,
  const unsigned int height, cudaStream_t stream);
