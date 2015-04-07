//
//  openGLhelpers.c
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "openGLhelpers.hpp"

#include "declareHostConstants.h"
#include "declareHostParameters.h"

//The steps are:
//1. Create an empty vertex buffer object (VBO)
//2. Register the VBO with Cuda
//3. Map the VBO for writing from Cuda
//4. Run Cuda kernel to modify the vertex positions
//5. Unmap the VBO
//6. Render the results using OpenGL

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
GLFWwindow* initGL( void )
{
	// Init GLFW
    glfwInit();
	// Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	
	// Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
//        return -1;
    }
	
	// Set the required callback functions
    glfwSetKeyCallback(window, key_callback);
	
	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
//        return -1;
    }
	
    int fbwidth, fbheight;
    glfwGetFramebufferSize(window,
                           &fbwidth,
                           &fbheight) ;
	// Define the viewport dimensions
    glViewport(0, 0, fbwidth, fbheight);
	
	return window;
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void createPBO(GLuint *PBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource **PBO_res)
{
	assert(PBO);
	assert(VAO);
	
	// create buffer object
	glGenVertexArrays(1,
					  VAO);
	glGenBuffers(1,
				 PBO);
	
	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER,
				 *PBO);
	// initialize buffer object
	unsigned int size = NUMBER_OF_ATOMS * 3 * sizeof(double);
	glBufferData(GL_ARRAY_BUFFER,
				 size,
				 0,
				 GL_DYNAMIC_DRAW);
	
	// Position attribute
	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(GLdouble), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(PBO_res,
								 *PBO,
								 CU_GRAPHICS_REGISTER_FLAGS_NONE);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Create CBO
////////////////////////////////////////////////////////////////////////////////
void createCBO(GLuint *CBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource **CBO_res)
{
	assert(CBO);
	assert(VAO);
	
	// create buffer object
	glGenBuffers(1,
				 CBO);
	
	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER,
				 *CBO);
	// initialize buffer object
	unsigned int size = NUMBER_OF_ATOMS * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER,
				 size,
				 0,
				 GL_DYNAMIC_DRAW);
	
	// Position attribute
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(1);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(CBO_res,
								 *CBO,
								 CU_GRAPHICS_REGISTER_FLAGS_NONE);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Render particles
////////////////////////////////////////////////////////////////////////////////
void renderParticles(GLuint *VAO,
					 Shader miShader)
{
	miShader.Use();
	
	// Render
	// Clear the colorbuffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1, 1, 1);
	
	float sigmax = 3. * 12. * kB * 20.e-6 / gs / muB / dBdz;
	
	glUniform1f(glGetUniformLocation(miShader.Program, "Tx"), sigmax);
	
	glBindVertexArray(*VAO);
	
	glDrawArrays(GL_POINTS, 0, NUMBER_OF_ATOMS);
	
	glBindVertexArray(0);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete BO
////////////////////////////////////////////////////////////////////////////////
void deleteBO(GLuint *BO,
			  GLuint *AO,
			  struct cudaGraphicsResource *BO_res)
{
	
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(BO_res);
	
	glDeleteVertexArrays(1, AO);
	glDeleteBuffers(1, BO);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void key_callback(GLFWwindow* window,
				  int key,
				  int scancode,
				  int action,
				  int mode)
{
	std::cout << key << std::endl;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

void computeFPS(GLFWwindow* window,
				int frameCount)
{
	float avgFPS = frameCount / glfwGetTime();
	
	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps", avgFPS);
	
	glfwSetWindowTitle( window, fps );
}