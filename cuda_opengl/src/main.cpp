#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "cpu_processor.h"
#include "driver/cuda_helper.h"
#include "driver/gpu_info.h"
#include "driver/interop.h"
#include "gpu_processor.h"
#include "raytrace.h"
#include "scene.h"
#include "utils.h"

#define USE_CPU

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

  unsigned int pow2_width = utils::nextPow2(width);
  unsigned int pow2_height = utils::nextPow2(height);
	interop->setSize(pow2_width, pow2_height);
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

  GLFWwindow* window;
  glfw_init(&window, 1024, 1024);

  // Gets back information about the GPUs.
  driver::GPUInfo gpu_info;
  cudaSetDevice(gpu_info.getCUDAGPU().device_id);

  // Logs information about the GPUs.
  std::cout << gpu_info.getProfile() << std::endl;

  // Parses selected scene using TinyObjLoader.
  //scene::Scene scene(argv[1]);
  scene::Scene scene("assets/cube.obj");
  std::cout << "uploading .obj scene to the GPU..." << std::endl;
#ifdef USE_CPU
  processor::CPUProcessor processor(scene, 1024, 1024);
#else
  GPUProcessor processor(scene, 1024, 1024);
#endif
  if (!processor.init())
  {
    std::cerr << "artracer: obj parsing failed.\n";
    std::cerr << "output: " << scene.error() << std::endl;
    return 1;
  }

  // Logs information about the GPUs, allows to see
  // how much memory is consummed by the obj scene.
  std::cout << gpu_info.getProfile() << std::endl;

	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamDefault);

  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);

  unsigned int pow2_width = utils::nextPow2(width);
  unsigned int pow2_height = utils::nextPow2(height);

  driver::Interop interop(pow2_width, pow2_height);

	glfwSetWindowUserPointer(window, &interop);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

  cudaError_t cuda_err;
	while (!glfwWindowShouldClose(window))
	{
		/*interop.getSize(pow2_width, pow2_height);
    cuda_err = interop.map(stream);

		cuda_err = raytrace(interop.getArray(), scene.getSceneData(),
      width, height, stream);
		cuda_err = interop.unmap(stream);

		interop.blit();
		interop.swap();*/

    processor.run();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	cudaDeviceReset();
  //scene.release();

	exit(EXIT_SUCCESS);
}