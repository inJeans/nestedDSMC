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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Other includes
#include "Shader.hpp"
#include "moveAtoms.cuh"

// Function prototypes
void computeFPS( GLFWwindow* window );

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

#endif /* defined(__nestedDSMC__main__) */
