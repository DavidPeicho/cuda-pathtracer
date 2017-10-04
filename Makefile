CC=g++
NVCC=nvcc
OBJS=main.o
CXXFLAGS=-Wall -Werror -Wextra -pedantic -std=c++17
NVCCFLAGS=-O3
LDFLAGS=-L/usr/local/cuda/lib
LDLIBS=-lcuda -lcudart -lcublas

BIN=artracer

VPATH=src

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^
	$(NVCC) $^ -o $@ $(LBLIBS)

clean:
	$(RM) $(BIN)
	$(RM) $(OBJS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -dc $<

.PHONY: clean all
