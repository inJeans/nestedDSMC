//
//  main.c
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "main.cuh"

long unsigned int frameCount = 0;

// The MAIN function, from here we start the application and run the game loop
int main()
{
    std::cout << "Starting GLFW context, OpenGL 3.3" << std::endl;
    // Init GLFW
	GLFWwindow* window = initGL( );
	
	Shader miShader("./src/shader.vert", "./src/shader.frag");
	
	// Create Vertex Array Buffer
	GLuint VAO;
	
	// Create Position Buffer Object
	struct cudaGraphicsResource *cudaPBOres;
	GLuint PBO;
	createPBO(&PBO,
			  &VAO,
			  &cudaPBOres);
	
	// Create Colour Buffer Object
    struct cudaGraphicsResource *cudaCBOres;
	GLuint CBO;
	createCBO(&CBO,
			  &VAO,
			  &cudaCBOres);
	
	curandState_t *d_rngStates;
	cudaMalloc( (void **)&d_rngStates, NUMBER_OF_ATOMS*sizeof(curandState_t) );
	
	double3 *d_vel;
	double3 *d_acc;
	
	cudaCalloc( (void **)&d_vel, NUMBER_OF_ATOMS, sizeof(double3) );
	cudaCalloc( (void **)&d_acc, NUMBER_OF_ATOMS, sizeof(double3) );
	
	h_initRNG(d_rngStates,
			  NUMBER_OF_ATOMS);
	
	h_generateInitialDist(&cudaPBOres,
						  d_vel,
						  d_acc,
						  NUMBER_OF_ATOMS,
						  d_rngStates);
	
    // Game loop
    while (!glfwWindowShouldClose(window))
    {
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();
		
		h_moveParticles(&cudaPBOres,
						d_vel,
						d_acc,
						1.e-5,
						NUMBER_OF_ATOMS);
		
		h_setParticleColour(d_vel,
							&cudaCBOres,
							20.e-6,
							NUMBER_OF_ATOMS);
		
		renderParticles(&VAO,
						miShader);
		
        // Swap the screen buffers
        glfwSwapBuffers(window);
		
		frameCount++;
		computeFPS(window,
				   frameCount);
    }
	
    // Properly de-allocate all resources once they've outlived their purpose
	deleteBO(&PBO,
			 &VAO,
			 cudaPBOres);
	deleteBO(&CBO,
			 &VAO,
			 cudaCBOres);
	
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
	
	cudaFree( d_vel );
	cudaFree( d_acc );
	
	cudaDeviceReset();
	
    return 0;
}