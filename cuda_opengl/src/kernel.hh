#pragma once

cudaError_t
kernel_launcher(cudaArray_const_t array, const int width, const int height, cudaEvent_t event, cudaStream_t stream);