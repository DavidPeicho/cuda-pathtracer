#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <iomanip>
#include <iostream>

#include "cpu_processor.h"
#include "driver/cuda_helper.h"
#include "driver/gpu_info.h"
#include "driver/interop.h"
#include "gpu_processor.h"
#include "raytrace.h"
#include "scene.h"
#include "utils.h"

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

void
glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
#ifdef USE_CPU
  processor::CPUProcessor* const processor =
    (processor::CPUProcessor* const)glfwGetWindowUserPointer(window);
#else
  processor::GPUProcessor* const processor =
    (processor::GPUProcessor* const)glfwGetWindowUserPointer(window);
#endif

	processor->setKeyState(key, action != GLFW_RELEASE);
}

void
glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
#ifdef USE_CPU
  processor::CPUProcessor* const processor =
    (processor::CPUProcessor* const)glfwGetWindowUserPointer(window);
#else
  processor::GPUProcessor* const processor =
    (processor::GPUProcessor* const)glfwGetWindowUserPointer(window);
#endif

	processor->setMoved(true);
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

  const int window_w = 1024;
  const int window_h = 1024;

  GLFWwindow* window;
  glfw_init(&window, window_w, window_h);

  // Parses selected scene using TinyObjLoader.
  //scene::Scene scene(argv[1]);
  scene::Scene scene("assets/cube-centered.obj");
  std::cout << "uploading .obj scene to the GPU..." << std::endl;
#ifdef USE_CPU
  processor::CPUProcessor processor(scene, 1024, 1024);
#else
  processor::GPUProcessor processor(scene, window_w, window_h);
#endif
  if (!processor.init())
  {
    std::cerr << "artracer: obj parsing failed.\n";
    std::cerr << "output: " << scene.error() << std::endl;
    return 1;
  }

	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(window, &width, &height);

	unsigned int pow2_width = utils::nextPow2(width);
	unsigned int pow2_height = utils::nextPow2(height);

	glfwSetWindowUserPointer(window, &processor);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);
	glfwSetKeyCallback(window, glfw_key_callback);
	glfwSetCursorPosCallback(window, glfw_mouse_callback);

  double last_time = 0.0;
  double curr_time = 0.0;
  double delta = 0.0;
  double elapsed = 0.0;

  cudaError_t cuda_err;
	while (!glfwWindowShouldClose(window))
	{
    curr_time = glfwGetTime();
    delta = curr_time - last_time;
    last_time = curr_time;

    if (elapsed >= 2.0)
    {
      std::cout << "\rartracer: FPS: " << std::fixed << std::setprecision(3) << (1.0 / delta);
      elapsed = 0.0;
    }

    processor.run();

		glfwSwapBuffers(window);
		glfwPollEvents();

		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

    processor.setMousePos(xpos, ypos);

		glfwSetCursorPos(window, width / 2, height / 2);
    elapsed += delta;
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	cudaDeviceReset();
	//scene.release();

	exit(EXIT_SUCCESS);
}