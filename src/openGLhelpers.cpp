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

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

bool keys[1024];


double firstx = 0.;
double firsty = 0.;

double xpos = 0.;
double ypos = 0.;

int fbwidth, fbheight;

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
	char fps[256];
	sprintf(fps, "CUDA DSMC - %i particles | %3.1f fps | T = %3.1f uK | <E> = %3.1f uK ", NUMBER_OF_ATOMS, 0., 20., 0.);
    GLFWwindow* window = glfwCreateWindow(800, 600, fps, NULL, NULL);
    glfwMakeContextCurrent(window);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
//        return -1;
    }
	
	// Set the required callback functions
    glfwSetKeyCallback(window,
					   key_callback);
	glfwSetScrollCallback(window,
						  scroll_callback);
//	glfwSetMouseButtonCallback(window,
//							   mouse_button_callback);
	
	// Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;
	// Initialize GLEW to setup the OpenGL Function pointers
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
//        return -1;
    }
	
//    int fbwidth, fbheight;
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
	
	// Camera/View transformation
	glm::mat4 view;
	view = camera.GetViewMatrix();
	
	// Projection
	glm::mat4 projection;
	projection = glm::perspective(camera.Zoom,
								  (float)fbwidth/(float)fbheight,
								  0.1f,
								  1000.0f);
	
	// Calculate the model matrix for each object and pass it to shader before drawing
	glm::mat4 model;
	
	// Get the uniform locations
	GLint modelLoc = glGetUniformLocation(miShader.Program,
										  "model");
	GLint viewLoc  = glGetUniformLocation(miShader.Program,
										  "view");
	GLint projLoc  = glGetUniformLocation(miShader.Program,
										  "projection");
	
	// Pass the matrices to the shader
	glUniformMatrix4fv(viewLoc,
					   1,
					   GL_FALSE,
					   glm::value_ptr(view));
	glUniformMatrix4fv(projLoc,
					   1,
					   GL_FALSE,
					   glm::value_ptr(projection));
	glUniformMatrix4fv(modelLoc,
					   1,
					   GL_FALSE,
					   glm::value_ptr(model));
	
	// Render
	// Clear the colorbuffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	
	float sigmax = 1. * 12. * h_kB * 20.e-6 / h_gs / h_muB / h_dBdz;
	
	glUniform1f(glGetUniformLocation(miShader.Program, "Tx"),
				sigmax);
	
	glBindVertexArray(*VAO);
	
	glPointSize(2.0f);
	glDrawArrays(GL_POINTS,
				 0,
				 NUMBER_OF_ATOMS);
	
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
// Moves/alters the camera positions based on user input
void Do_Movement(GLfloat deltaTime)
{
	// Camera controls
	if(keys[GLFW_KEY_W] || keys[GLFW_KEY_UP])
	{
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}
	else if(keys[GLFW_KEY_S] || keys[GLFW_KEY_DOWN])
	{
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}
	else if(keys[GLFW_KEY_A] || keys[GLFW_KEY_LEFT])
	{
		camera.ProcessKeyboard(LEFT, deltaTime);
	}
	else if(keys[GLFW_KEY_D] || keys[GLFW_KEY_RIGHT])
	{
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}
	
	return;
}

void key_callback(GLFWwindow* window,
				  int key,
				  int scancode,
				  int action,
				  int mode)
{
	std::cout << key << std::endl;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	
	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
			keys[key] = true;
		else if (action == GLFW_RELEASE)
			keys[key] = false;
	}
}

//void mouse_button_callback(GLFWwindow* window,
//						   int button,
//						   int action,
//						   int mods)
//{
//	
//	
//	if(button == GLFW_MOUSE_BUTTON_LEFT &&
//		action == GLFW_PRESS)
//	{
//		glfwGetCursorPos(window, &firstx, &firsty);
//		std::cout << "First press x = " << firstx << " xpos = " << xpos << std::endl;
//	}
//	
//	if(button == GLFW_MOUSE_BUTTON_LEFT &&
//	   action == GLFW_RELEASE)
//	{
//		glfwGetCursorPos(window, &xpos, &ypos);
//		
//		GLfloat xoffset = -xpos + firstx;
//		GLfloat yoffset = ypos - firsty;
//
//		std::cout << "Release x = " << xpos << " first x = " << firstx << std::endl;
//		
//		firstx = 0.;
//		firsty = 0.;
//		
//		xpos = 0.;
//		ypos = 0.;
//		
//		camera.ProcessMouseMovement(xoffset,
//									yoffset);
//	}
//	
//	return;
//}

void scroll_callback(GLFWwindow* window,
					 double xoffset,
					 double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}

void computeFPS(GLFWwindow* window,
				int         numberOfAtoms,
				double      T,
				double      E,
				int         frameCount)
{
	float avgFPS = frameCount / glfwGetTime();
	
	char fps[256];
	sprintf(fps, "CUDA DSMC - %i particles | %3.1f fps | T = %3.1f uK | <E> = %3.1f uK ", numberOfAtoms, avgFPS, T*1.e6, E*1.e6/h_kB);
	
	glfwSetWindowTitle( window, fps );
}

double3* mapCUDAVBOd3(struct cudaGraphicsResource **cudaVBOres)
{
	// Map OpenGL buffer object for writing from CUDA
	double3 *d_ptr;
	cudaGraphicsMapResources(1,
							 cudaVBOres,
							 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_ptr,
										 &num_bytes,
										 *cudaVBOres);

	return d_ptr;
}

float4* mapCUDAVBOf4(struct cudaGraphicsResource **cudaVBOres)
{
	// Map OpenGL buffer object for writing from CUDA
	float4 *d_ptr;
	cudaGraphicsMapResources(1,
							 cudaVBOres,
							 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_ptr,
										 &num_bytes,
										 *cudaVBOres);
	
	return d_ptr;
}

void unmapCUDAVBO(struct cudaGraphicsResource **cudaVBOres)
{
	//Unmap buffer object
	cudaGraphicsUnmapResources(1,
							   cudaVBOres,
							   0);
	
	return;
}