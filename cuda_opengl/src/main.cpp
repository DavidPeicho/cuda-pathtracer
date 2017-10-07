#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "driver/cuda_helper.h"
#include "driver/gpu_info.h"
#include "driver/interop.h"
#include "kernel.hh"
#include "scene.h"

static void
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
  if (argc < 2)
  {
    std::cerr << "artracer: missing scene argument.\n";
    std::cerr << "usage: artracer [OBJ SCENE]" << std::endl;
    return 1;
  }

  // Parses selected scene using TinyObjLoader.
  scene::Scene scene("assets/cube.obj");
  if (!scene.ready())
  {
    std::cerr << "artracer: obj parsing failed.\n";
    std::cerr << "output: " << scene.error() << std::endl;
    return 1;
  }

  // Gets back information about the GPUs.
  driver::GPUInfo gpu_info;
  cudaSetDevice(gpu_info.getCUDAGPU().device_id);

  const unsigned int resolution_width = 512;
  const unsigned int resolution_height = 512;

	GLFWwindow* window;
	glfw_init(&window, 1024, 1024);

	cudaStream_t stream;
	cudaEvent_t  event;

	cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
	cudaEventCreateWithFlags(&event, cudaEventBlockingSync);

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  driver::Interop interop(width, height);

	glfwSetWindowUserPointer(window, &interop);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

  cudaError_t cuda_err;
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