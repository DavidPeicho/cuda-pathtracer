#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <iomanip>
#include <iostream>

#include "cpu_processor.h"
#include "driver/cuda_helper.h"
#include "driver/gpu_info.h"
#include "driver/interop.h"
#include "gpu_processor.h"

#include "scene.h"
#include "utils.h"

static void
glfw_init(GLFWwindow** window, const int width, const int height)
{
	if (!glfwInit()) exit(EXIT_FAILURE);

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
  processor::GPUProcessor* const processor =
    (processor::GPUProcessor* const)glfwGetWindowUserPointer(window);

  auto& interop = processor->getInterop();

	interop.setSize(width, height);
}

void
glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  processor::GPUProcessor* const processor =
    (processor::GPUProcessor* const)glfwGetWindowUserPointer(window);

  if (key < 0 || key >= 1024) return;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

	processor->setKeyState(key, action != GLFW_RELEASE);
}

void
glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
  processor::GPUProcessor* const processor =
    (processor::GPUProcessor* const)glfwGetWindowUserPointer(window);

	processor->setMoved(true);
  processor->setMousePos(xpos, ypos);
}

int
main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "artracer: missing scene argument.\n";
    std::cerr << "usage: artracer [OBJ SCENE]" << std::endl;
    //return 1;
  }

  const int window_w = 960;
  const int window_h = 540;

  GLFWwindow* window;
  glfw_init(&window, window_w, window_h);

  // Parses selected scene using TinyObjLoader.
  scene::Scene scene("assets/wooden_hut_hill.scene");
  //scene::Scene scene("assets/crate_land.scene");
  //scene::Scene scene("assets/sss_crate.scene");
  std::cout << "uploading .obj scene to the GPU..." << std::endl;

  processor::GPUProcessor processor(scene, window_w, window_h);

	int width = 0;
	int height = 0;
	glfwGetFramebufferSize(window, &width, &height);

	unsigned int pow2_width = utils::nextPow2(width);
	unsigned int pow2_height = utils::nextPow2(height);

  glfwSetWindowUserPointer(window, &processor);
	glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);
	glfwSetKeyCallback(window, glfw_key_callback);
	glfwSetCursorPosCallback(window, glfw_mouse_callback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

  try
  {
    if (!processor.init())
    {
      std::cerr << "artracer: obj parsing failed.\n";
      std::cerr << "output: " << scene.error() << std::endl;
      return 1;
    }
  } catch (std::exception e)
  {
    std::cerr << "artracer: exception: " << std::endl;
    std::cerr << e.what() << std::endl;
    return 1;
  }

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

    if (elapsed >= 5.0)
    {
      std::cout << "artracer: FPS: " << std::fixed << std::setprecision(3) << (1.0 / delta) << std::endl;
      elapsed = 0.0;
    }

    processor.run(delta);

		glfwSwapBuffers(window);
		glfwPollEvents();

		glfwSetCursorPos(window, width / 2, height / 2);
    elapsed += delta;
	}

  //cudaDeviceSynchronize();
  //scene.release();

	glfwDestroyWindow(window);
	glfwTerminate();

	cudaDeviceReset();

	exit(EXIT_SUCCESS);
}