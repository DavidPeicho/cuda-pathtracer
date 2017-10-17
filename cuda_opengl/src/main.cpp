#ifdef _WIN64
# include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <glm/common.hpp>
#include <glm/glm.hpp>

#include <iostream>

#include "driver/cuda_helper.h"
#include "driver/gpu_info.h"
#include "driver/interop.h"
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
	driver::Interop* const interop =
		(driver::Interop* const)glfwGetWindowUserPointer(window);

	interop->setKeyState(key, action != GLFW_RELEASE);
}

void
glfw_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	driver::Interop* const interop =
		(driver::Interop* const)glfwGetWindowUserPointer(window);
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

	// Gets back information about the GPUs.
	driver::GPUInfo gpu_info;
	cudaSetDevice(gpu_info.getCUDAGPU().device_id);

	// Logs information about the GPUs.
	std::cout << gpu_info.getProfile() << std::endl;

	// Parses selected scene using TinyObjLoader.
	//scene::Scene scene(argv[1]);
	scene::Scene scene(argv[1]);
	if (!scene.ready())
	{
		std::cerr << "artracer: obj parsing failed.\n";
		std::cerr << "output: " << scene.error() << std::endl;
		return 1;
	}
	std::cout << "uploading .obj scene to the GPU..." << std::endl;
	scene.upload();

	// Logs information about the GPUs, allows to see
	// how much memory is consummed by the obj scene.
	std::cout << gpu_info.getProfile() << std::endl;

	GLFWwindow* window;
	glfw_init(&window, 1024, 1024);

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
	glfwSetKeyCallback(window, glfw_key_callback);
	glfwSetCursorPosCallback(window, glfw_mouse_callback);


	cudaError_t cuda_err;
	glm::vec3 offset = glm::vec3(0.0f);
	glm::vec2 angle = glm::vec2(0.0f);
	glm::vec3 dir_offset = glm::vec3(0.0f);

	glm::vec3* temporal_framebuffer;
	cudaMalloc(&temporal_framebuffer, width * height * sizeof(glm::vec3));

	while (!glfwWindowShouldClose(window))
	{
		interop.getSize(pow2_width, pow2_height);
		cuda_err = interop.map(stream);

		cuda_err = raytrace(interop.getArray(), scene.getDevicePointer(), width, height, stream, offset, dir_offset, temporal_framebuffer);
		cuda_err = interop.unmap(stream);

		interop.blit();
		interop.swap();

		glfwSwapBuffers(window);
		glfwPollEvents();

		if (interop.isKeyPressed(GLFW_KEY_Z))
			offset.z++;
		if (interop.isKeyPressed(GLFW_KEY_S))
			offset.z--;
		if (interop.isKeyPressed(GLFW_KEY_D))
			offset.x++;
		if (interop.isKeyPressed(GLFW_KEY_Q))
			offset.x--;
		if (interop.isKeyPressed(GLFW_KEY_SPACE))
			offset.y++;
		if (interop.isKeyPressed(GLFW_KEY_LEFT_CONTROL))
			offset.y--;

		//std::cout << offset.x << " " << offset.y << " " << offset.z << std::endl;

		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		angle.x += 0.03f * float(width / 2 - xpos);
		angle.y += 0.03f * float(height / 2 - ypos);

		dir_offset = glm::vec3(
			cos(angle.y) * sin(angle.x),
			sin(angle.y),
			cos(angle.y) * cos(angle.x)
		);

		dir_offset = glm::normalize(dir_offset);

		glfwSetCursorPos(window, width / 2, height / 2);
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	cudaFree(temporal_framebuffer);

	cudaDeviceReset();
	//scene.release();

	exit(EXIT_SUCCESS);
}