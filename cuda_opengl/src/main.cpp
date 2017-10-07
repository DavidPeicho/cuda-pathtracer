#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include <stdlib.h>
#include <iostream>
#include <stdbool.h>

#include <cuda_gl_interop.h>

#include "interop.h"

#include "kernel.hh"

static
void
glfw_init(GLFWwindow** window, const int width, const int height)
{
	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	*window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);

	if (*window == NULL)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(*window);

	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	glfwSwapInterval(0);

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
}

static
void
glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
	driver::Interop* const interop =
    (driver::Interop* const)glfwGetWindowUserPointer(window);

	interop->setSize(width, height);
}

int
main(int argc, char* argv[])
{
	GLFWwindow* window;

	glfw_init(&window, 1024, 1024);

	cudaError_t cuda_err;

	int gl_device_id;
	unsigned int gl_device_count;
	cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll);

	int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
	cudaSetDevice(cuda_device_id);

	struct cudaDeviceProp props;

	cudaGetDeviceProperties(&props, gl_device_id);
	std::cout << "GL   : " <<  props.name << " -> " << props.multiProcessorCount << std::endl;

	cudaGetDeviceProperties(&props, cuda_device_id);
	std::cout << "CUDA   : " << props.name << " -> " << props.multiProcessorCount << std::endl;

	cudaStream_t stream;
	cudaEvent_t  event;

	cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
	cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  driver::Interop interop(width, height);

	glfwSetWindowUserPointer(window, &interop);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

	while (!glfwWindowShouldClose(window))
	{
		cudaArray_t cuda_array;

		interop.getSize(width, height);
    cuda_err = interop.map(stream);

		cuda_err = kernel_launcher(interop.getArray(), width, height, event, stream);
		cuda_err = interop.unmap(stream);

		interop.blit();
		interop.swap();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	cudaDeviceReset();

	exit(EXIT_SUCCESS);
}