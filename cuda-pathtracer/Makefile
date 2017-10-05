CC=g++
NVCC=nvcc
NVCCFLAGS=-c --compiler-options='-Wall -Wextra -Werror -g'

BUILD_DIR=./build/

BIN=artracer

OBJS_CU=test.o
OBJS=main.o

VPATH=src

all: $(BIN)

$(BIN): $(OBJS) $(OBJS_CU)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $^ -o $@
	mv $^ $(BUILD_DIR)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	$(RM) $(BIN) a.out
	$(RM) $(BUILD_DIR)

.PHONY: clean all
