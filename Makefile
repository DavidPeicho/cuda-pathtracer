SLN_DIR=cuda_opengl

CC=g++

INC=-I./$(SLN_DIR)/3rd_party/include

NVCC=nvcc
NVCCFLAGS=-std=c++11 -G -g $(INC) -c --compiler-options='-Wall -Wextra'

LDLIBS= -lglfw
LDFLAGS= -L./lib

BUILD_DIR=./build/

BIN=artracer

OBJS_CU=raytrace.o
OBJS=gpu_info.o interop.o glad.o main.o scene.o utils.o \
     cpu_processor.o gpu_processor.o material_loader.o

DIRS=src src/driver 3rd_party
VPATH=$(SLN_DIR):$(foreach d, $(DIRS), :$(SLN_DIR)/$d)

all: $(BIN)

$(BIN): $(OBJS) $(OBJS_CU)
	#mkdir -p $(BUILD_DIR)
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
