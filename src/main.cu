//
//  main.c
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "main.cuh"

int NUMBER_OF_ATOMS = 1e3;
int frameCount = 0;

// The MAIN function, from here we start the application and run the game loop
int main()
{
    std::cout << "Starting GLFW context, OpenGL 3.3" << std::endl;
    // Init GLFW
	GLFWwindow* window = initGL( );;
	
	Shader miShader("./src/shader.vert", "./src/shader.frag");
	
    struct cudaGraphicsResource *cudaVBOres;
	// Create VBO
	GLuint VBO, VAO;
	createVBO(&VBO,
			  &VAO,
			  &cudaVBOres);
	
	GLfloat oldValue = glfwGetTime();
	
	double dt = 0.;
	
	curandState_t *d_rngStates;
	cudaMalloc( (void **)&d_rngStates, NUMBER_OF_ATOMS*sizeof(curandState_t) );
	
	double3 *d_vel;
	
	cudaCalloc( (void **)&d_vel, NUMBER_OF_ATOMS, sizeof(double3) );
	
	h_initRNG(d_rngStates,
			  NUMBER_OF_ATOMS);
	
	h_generateInitialDist(&cudaVBOres,
						  d_vel,
						  NUMBER_OF_ATOMS,
						  d_rngStates);
	
    // Game loop
    while (!glfwWindowShouldClose(window))
    {
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();
		
		GLfloat timeValue = glfwGetTime();
		
		dt = timeValue - oldValue;
		
		moveParticles(&cudaVBOres,
					  d_vel,
					  dt,
					  NUMBER_OF_ATOMS);
		
		renderParticles(&VAO,
						miShader);
		
        // Swap the screen buffers
        glfwSwapBuffers(window);
		
		frameCount++;
		computeFPS(window,
				   frameCount);
		
		oldValue = timeValue;
    }
	
    // Properly de-allocate all resources once they've outlived their purpose
	deleteVBO(&VBO,
			  &VAO,
			  cudaVBOres);
	
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
	
	cudaFree( d_vel );
	
	cudaDeviceReset();
	
    return 0;
}