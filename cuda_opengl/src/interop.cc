#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <iostream>

#include "interop.hh"

struct interop
{
	int count;
	int index;

	int width;
	int height;

	GLuint* fb;
	GLuint* rb;

	cudaGraphicsResource_t* cgr;
	cudaArray_t* ca;
};

struct interop*
create(const int fbo_count)
{
	struct interop* const interop = new struct interop();

	interop->count = fbo_count;
	interop->index = 0;

	interop->fb = new GLuint[fbo_count]();
	interop->rb = new GLuint[fbo_count]();
	interop->cgr = new cudaGraphicsResource_t[fbo_count]();
	interop->ca = new cudaArray_t[fbo_count]();

	glCreateRenderbuffers(fbo_count, interop->rb);

	glCreateFramebuffers(fbo_count, interop->fb);

	for (int i = 0; i < fbo_count; i++)
	{
		glNamedFramebufferRenderbuffer(interop->fb[i],
			GL_COLOR_ATTACHMENT0,
			GL_RENDERBUFFER,
			interop->rb[i]);
	}

	return interop;
}


void
clean(struct interop* const interop)
{
	cudaError_t cuda_err;

	for (int i = 0; i < interop->count; i++)
	{
		if (interop->cgr[i] != NULL)
			cuda_err = cudaGraphicsUnregisterResource(interop->cgr[i]);
	}

	glDeleteRenderbuffers(interop->count, interop->rb);

	glDeleteFramebuffers(interop->count, interop->fb);

	delete[] interop->fb;
	delete[] interop->rb;
	delete[] interop->cgr;
	delete[] interop->ca;

	delete interop;
}

cudaError_t
set_size(struct interop* const interop, const int width, const int height)
{
	cudaError_t cuda_err = cudaSuccess;

	interop->width = width;
	interop->height = height;

	for (int i = 0; i < interop->count; i++)
	{
		if (interop->cgr[i] != NULL)
			cudaGraphicsUnregisterResource(interop->cgr[i]);

		glNamedRenderbufferStorage(interop->rb[i], GL_RGBA8, width, height);

		cudaGraphicsGLRegisterImage(&interop->cgr[i], interop->rb[i], GL_RENDERBUFFER,
			cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
	}

	cudaGraphicsMapResources(interop->count, interop->cgr, 0);

	for (int index = 0; index < interop->count; index++)
	{
		cudaGraphicsSubResourceGetMappedArray(&interop->ca[index],
			interop->cgr[index],
			0, 0);
	}

	cudaGraphicsUnmapResources(interop->count, interop->cgr, 0);

	return cuda_err;
}

void
interop_size_get(struct interop* const interop, int& const width, int& const height)
{
	width = interop->width;
	height = interop->height;
}

cudaError_t
map_resource(struct interop* const interop, cudaStream_t stream)
{
	return cudaGraphicsMapResources(1, &interop->cgr[interop->index], stream);
}

cudaError_t
unmap_resource(struct interop* const interop, cudaStream_t stream)
{
	return cudaGraphicsUnmapResources(1, &interop->cgr[interop->index], stream);
}

cudaError_t
map_array(struct interop* const interop)
{
	cudaError_t cuda_err;

	cuda_err = cudaGraphicsSubResourceGetMappedArray(&interop->ca[interop->index],
		interop->cgr[interop->index],
		0, 0);
	return cuda_err;
}

cudaArray_const_t
get_array(struct interop* const interop)
{
	return interop->ca[interop->index];
}

int
get_interop_index(struct interop* const interop)
{
	return interop->index;
}


void
swap(struct interop* const interop)
{
	interop->index = (interop->index + 1) % interop->count;
}

void
clear(struct interop* const interop)
{
	const GLfloat clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glClearNamedFramebufferfv(interop->fb[interop->index], GL_COLOR, 0, clear_color);
}

void
blit(struct interop* const interop)
{
	glBlitNamedFramebuffer(interop->fb[interop->index], 0, 0, 0, interop->width, interop->height, 0, interop->height, interop->width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}