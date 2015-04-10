//
//  main.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__main__
#define __nestedDSMC__main__

#include <stdio.h>
#include <iostream>
#include <math.h>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

// Other includes
#include "defineHostParameters.h"
#include "defineHostConstants.h"
#include "Shader.hpp"
#include "setUp.cuh"
#include "moveAtoms.cuh"
#include "numberCrunch.cuh"
#include "openGLkernels.cuh"
#include "openGLhelpers.hpp"

// Function prototypes

#define cudaCalloc(A, B, C) \
do { \
cudaError_t __cudaCalloc_err = cudaMalloc(A, (B)*C); \
if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, (B)*C); \
} while (0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif /* defined(__nestedDSMC__main__) */
