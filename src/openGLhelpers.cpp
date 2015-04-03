//
//  openGLhelpers.c
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "openGLhelpers.hpp"

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
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	
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
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *VBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource **VBO_res)
{
	assert(VBO);
	assert(VAO);
	
	// create buffer object
	glGenVertexArrays(1,
					  VAO);
	glGenBuffers(1,
				 VBO);
	
	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER,
				 *VBO);
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
	cudaGraphicsGLRegisterBuffer(VBO_res,
								 *VBO,
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
	
	// Update the uniform color
	GLfloat timeValue = glfwGetTime();
	GLfloat greenValue = (sin(timeValue) / 2) + 0.5;
	glUniform4f(glGetUniformLocation(miShader.Program, "ourColor"), 0.0f, greenValue, 0.0f, 1.0f);
	
	glBindVertexArray(*VAO);
	glPointSize(1.0f);//set point size to 10 pixels
	glDrawArrays(GL_POINTS, 0, NUMBER_OF_ATOMS);
	glBindVertexArray(0);
	
	return;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *VBO,
			   GLuint *VAO,
			   struct cudaGraphicsResource *VBO_res)
{
	
	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(VBO_res);
	
	glDeleteVertexArrays(1, VAO);
	glDeleteBuffers(1, VBO);
	
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