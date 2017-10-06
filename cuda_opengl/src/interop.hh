#pragma once

#include <cuda_runtime.h>

struct interop*
create(const int fbo_count);

void
clean(struct interop* const interop);

cudaError_t
set_size(struct interop* const interop, const int width, const int height);

void
interop_size_get(struct interop* const interop, int& const width, int& const height);

cudaError_t
map_resource(struct interop* const interop, cudaStream_t stream);
 
cudaError_t
unmap_resource(struct interop* const interop, cudaStream_t stream);

cudaError_t
map_array(struct interop* const interop);

cudaArray_const_t
get_array(struct interop* const interop);

int
get_interop_index(struct interop* const interop);

void
swap(struct interop* const interop);

void
clear(struct interop* const interop);

void
blit(struct interop* const interop);