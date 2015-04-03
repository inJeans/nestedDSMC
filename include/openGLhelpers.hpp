//
//  openGLhelpers.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 03/04/2015.
//
//

#ifndef __nestedDSMC__openGLhelpers__
#define __nestedDSMC__openGLhelpers__

#include <iostream>
#include <assert.h>

#include <math.h>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Shader.hpp"
#include "systemParameters.h"

GLFWwindow* initGL( void );

void createVBO(GLuint *VBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource **VBO_res);

void renderParticles(GLuint *VAO,
					 Shader miShader);

void deleteVBO(GLuint *VBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource *VBO_res);

void computeFPS(GLFWwindow* window,
				int frameCount);

void key_callback(GLFWwindow* window,
				  int key,
				  int scancode,
				  int action,
				  int mode);

#endif /* defined(__nestedDSMC__openGLhelpers__) */