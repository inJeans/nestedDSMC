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

// GLM Mathematics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "declareHostConstants.h"
#include "declareHostParameters.h"

#include "Shader.hpp"
#include "Camera.hpp"

GLFWwindow* initGL( void );

void createPBO(GLuint *PBO,
			   GLuint *PAO,
			   struct cudaGraphicsResource **PBO_res);

void createCBO(GLuint *CBO,
			   GLuint *CAO,
			   struct cudaGraphicsResource **CBO_res);

void renderParticles(GLuint *VAO,
					 Shader miShader);

void deleteBO(GLuint *BO,
			  GLuint *AO,
			  struct cudaGraphicsResource *BO_res);

void Do_Movement(GLfloat deltaTime);

void key_callback(GLFWwindow* window,
				  int key,
				  int scancode,
				  int action,
				  int mode);

//void mouse_button_callback(GLFWwindow* window,
//						   int button,
//						   int action,
//						   int mods);

void scroll_callback(GLFWwindow* window,
					 double xoffset,
					 double yoffset);

void computeFPS(GLFWwindow* window,
				int         numberOfAtoms,
				double      T,
				double      E,
				int         frameCount);

double3* mapCUDAVBOd3(struct cudaGraphicsResource **cudaVBOres);
float4*  mapCUDAVBOf4(struct cudaGraphicsResource **cudaVBOres);

void unmapCUDAVBO(struct cudaGraphicsResource **cudaVBOres);

#endif /* defined(__nestedDSMC__openGLhelpers__) */