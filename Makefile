#  Makefile.sh
#  nestedDSMC
#
#  Created by Christopher Watkins on 1/08/2014.
#  Copyright (c) 2014 WIJ. All rights reserved.

# Colour coding the makfile :-)
NO_COLOR=\033[0m
OK_COLOR=\033[32m
ERROR_COLOR=\033[31m
WARN_COLOR=\033[35m

OK_STRING=$(OK_COLOR)[OK]$(NO_COLOR)
ERROR_STRING=$(ERROR_COLOR)[ERRORS]$(NO_COLOR)
WARN_STRING=$(WARN_COLOR)[WARNINGS]$(NO_COLOR)

# Common binaries
GCC   ?= g++
CLANG ?= /usr/bin/clang++

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin) #If building on an OSX system
	CUDA_PATH= /Developer/NVIDIA/CUDA-7.0
	CUDA_INC = -I/Developer/NVIDIA/CUDA-7.0/include
	CUDA_LIB = -L/Developer/NVIDIA/CUDA-7.0/lib
	NVCC     = $(CUDA_PATH)/bin/nvcc -ccbin $(CLANG)
	INCLUDE = -I /usr/local/hdf5/include -I /usr/local/Cellar/glew/1.11.0/include/ -I/usr/local/Cellar/glm/0.9.6.1/include -I/Developer/NVIDIA/cub-1.4.0/
	LIB = -L/usr/local/hdf5/lib -L/usr/local/Cellar/glew/1.11.0/lib/ -L/System/Library/Frameworks/OpenGL.framework/Libraries
else                     #If building on a Linux system
	CUDA_INC = -I/usr/local/cuda-5.5/include
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)  #If 64 bit
		CUDA_LIB = -L/usr/local/cuda-5.5/lib64
	else
		CUDA_LIB = -L/usr/local/cuda-5.5/lib
	endif
	NVCC = nvcc
	INCLUDE = -I /usr/local/hdf5/1.8.5/include
	LIB = -L /usr/local/hdf5/1.8.5/lib
endif

#NVCCFLAGS = -m 64 --relocatable-device-code=true -arch=compute_30 -code=sm_30,compute_30
#NVCCFLAGS = -m 64 -arch=compute_30 -code=sm_30,compute_30
NVCCFLAGS = -m 64 -arch=sm_35

BUILDDIR = bin/
OBJDIR   = $(BUILDDIR)obj/
SRCDIR   = src/

EXEC = $(BUILDDIR)CUDADSMC
TESTMOTION = $(BUILDDIR)testMotion
TESTCOLLISIONS = $(BUILDDIR)testCollisions

INCLUDE += -I include/

all: $(EXEC)

debug: NVCCFLAGS += -g -G
#debug: clean
debug: $(EXEC)

profile: NVCCFLAGS += -pg -lineinfo
profile: $(EXEC)

$(EXEC): $(addprefix $(OBJDIR), gpuCode.o main.o setUp.o moveAtoms.o collideAtoms.o magneticField.o numberCrunch.o openGLKernels.o openGLhelpers.o cudaHelpers.o shader.o camera.o)
	@echo 'Building file: $@'
	@echo 'Invoking: NVCC Linker'
	clang++ -o $@ $(INCLUDE) $^ $(LIB) $(CUDA_LIB) -lc++ -lcudart -lcudadevrt -lcurand -lglfw3 -lGLEW -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo
	@echo "Finished building: $@ $(OK_STRING)"
	@echo ' '

$(OBJDIR)gpuCode.o: $(addprefix $(OBJDIR), main.o setUp.o moveAtoms.o collideAtoms.o magneticField.o numberCrunch.o openGLKernels.o cudaHelpers.o)
	@echo 'Linking device object files: $@'
	@echo 'Invoking: NVCC Linker'
	$(NVCC) $(NVCCFLAGS) -dlink -o $@ $^
	@echo "Finished linking: $@ $(OK_STRING)"
	@echo ' '

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: clang Compiler'
	clang++ $(INCLUDE) $(CUDA_INC) -o $@ -c $?
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '

$(OBJDIR)%.o: $(SRCDIR)%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC) $(NVCCFLAGS) -x cu $(INCLUDE) -o $@ -dc $? -D CUDA7 -D CUB_CDP
	@echo "Finished building: $< $(OK_STRING)"
	@echo ' '

run:
	make
	{ time ./$(BUILDDIR)CUDADSMC ; } | tee output.txt
#	/usr/local/hdf5/bin/h5import "./src/main.cu" -c "./hdfimportconfig/h5import-main.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "./src/evolveSpin.cu" -c "./hdfimportconfig/h5import-evolveSpin.conf" -o "mmDataMCWF.h5"
#	/usr/local/hdf5/bin/h5import "Makefile" -c "./hdfimportconfig/h5import-makefile.conf" -o "mmDataMCWF.h5"
#	@echo "" >> output.txt
#	@echo "----------------------------------------" >> output.txt
#	@echo "" >> output.txt
	tput bel

memcheck:
	{ time cuda-memcheck ./$(BUILDDIR)CUDADSMC ; } | tee output.txt
	@echo "" >> output.txt
	@echo "----------------------------------------" >> output.txt
	@echo "" >> output.txt
	tput bel

clean:
	rm -rf $(OBJDIR)*.o $(BUILDDIR)CUDADSMC
