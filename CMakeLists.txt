project(Artracer LANGUAGES CXX)

cmake_minimum_required(VERSION 3.0)
include(ExternalProject)

find_package(CUDA QUIET REQUIRED)

set(GLFW_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/glfw)

ExternalProject_Add(GLFW
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${GLFW_INSTALL_LOCATION}" "-DBUILD_SHARED_LIBS=ON"
)

set(TARGET artracer)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11"
)
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS} -O2 -Wno-deprecated-gpu-targets"
)

set(SLN_DIR cuda_opengl)
set(GLFW_DIR glfw)

include_directories(
  ${SLN_DIR}/3rd_party/include
  ${SLN_DIR}/include
  ${GLFW_INSTALL_LOCATION}/include
)

link_directories(${GLFW_INSTALL_LOCATION}/lib)

set(SRC
    ${SLN_DIR}/src/driver/gpu_info.cpp
    ${SLN_DIR}/src/driver/interop.cpp
    ${SLN_DIR}/src/glad.cpp
    ${SLN_DIR}/src/gpu_processor.cpp
    ${SLN_DIR}/src/gui/gui_manager.cpp
    ${SLN_DIR}/src/gui/imgui.cpp
    ${SLN_DIR}/src/gui/imgui_draw.cpp
    ${SLN_DIR}/src/gui/imgui_impl_glfw_gl3.cpp
    ${SLN_DIR}/src/main.cpp
    ${SLN_DIR}/src/material_loader.cpp
    ${SLN_DIR}/src/scene/scene.cpp
    ${SLN_DIR}/src/shaders/raytrace.cu
    ${SLN_DIR}/src/texture_utils.cpp
    ${SLN_DIR}/src/utils.cpp
)

CUDA_ADD_EXECUTABLE(${TARGET} ${SRC})

add_dependencies(${TARGET} GLFW)

target_link_libraries(${TARGET} glfw)

set_property(
  TARGET ${TARGET}
  PROPERTY CUDA_SEPARABLE_COMPILATION ON
)

