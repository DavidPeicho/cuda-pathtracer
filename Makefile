CC = g++
SLN_DIR = cuda_opengl
# ASSET = $(SLN_DIR)/assets/crate_land.scene

INC_DIR = 3rd_party/include src src/shaders src/driver src/scene src/gui
INC=$(foreach d, $(INC_DIR), -I./$(SLN_DIR)/$d )

NVCC=nvcc
NVCCFLAGS=-arch=sm_21 -std=c++11 -g $(INC) -Wno-deprecated-gpu-targets -c --compiler-options='-Wall -Wextra'

GLFW_DIR=./glfw
LIB_DIR=$(GLFW_DIR)/build/src

LDLIBS= -lglfw
LDFLAGS= -L$(LIB_DIR)

BUILD_DIR=./build/

BIN=artracer

OBJS_CU=raytrace.o
OBJS= glad.o \
      gpu_info.o \
      gpu_processor.o \
      gui_manager.o \
      imgui.o \
      imgui_draw.o \
      imgui_impl_glfw_gl3.o \
      interop.o \
      main.o \
      material_loader.o \
      scene.o \
      texture_utils.o \
      utils.o

DIRS=src src/driver 3rd_party src/shaders src/scene src/gui
VPATH=$(SLN_DIR):$(foreach d, $(DIRS), :$(SLN_DIR)/$d)

all: $(BIN)

$(BIN): $(OBJS) $(OBJS_CU)
	#git clone https://github.com/glfw/glfw.git || true
	#cd $(GLFW_DIR) && mkdir -p build && cd build && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug .. && make
	$(NVCC) $^ -o $@ $(LDFLAGS) $(LDLIBS)
	#mv $^ $(BUILD_DIR)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	$(RM) $(BIN) $(OBJS) $(OBJS_CU)
	$(RM) -r $(BUILD_DIR)

.PHONY: clean all
