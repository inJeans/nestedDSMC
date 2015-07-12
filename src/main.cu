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
	cudaDeviceReset();
	
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
	std::cout << "rng Malloc: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
	
	double3 *d_vel;
	double3 *d_acc;
	
	double *d_sigvrmax;
	
	int2 *d_cellStartEnd;
	
	int *d_atomID;
	int *d_cellID;
	int *d_numberOfCollisions;
	int *d_atomcount;
	
	cudaCalloc( (void **)&d_vel, NUMBER_OF_ATOMS, sizeof(double3) );
	cudaCalloc( (void **)&d_acc, NUMBER_OF_ATOMS, sizeof(double3) );
	
	cudaCalloc( (void **)&d_sigvrmax, 1, sizeof(double) );
	
	cudaCalloc( (void **)&d_cellStartEnd, 1, sizeof(int2) );
	
	cudaCalloc( (void **)&d_atomID, NUMBER_OF_ATOMS, sizeof(int) );
	cudaCalloc( (void **)&d_cellID, NUMBER_OF_ATOMS, sizeof(int) );
	cudaCalloc( (void **)&d_numberOfCollisions, 1, sizeof(int) );
	cudaCalloc( (void **)&d_atomcount, 1, sizeof(int) );
	
	double sigma = 8.*h_pi*h_a*h_a;
	double sigvrmax = sqrt(16.*h_kB*20.e-6/(h_pi*h_mRb))*sigma;
//	double sigvrmax = 0.;
	
	deviceMemset<<<1,1>>>(d_sigvrmax,
						  sigvrmax,
						  1 );
	
	deviceMemset<<<1,1>>>(d_cellStartEnd,
						  make_int2(0, NUMBER_OF_ATOMS-1),
						  1 );
	
	deviceMemset<<<NUMBER_OF_ATOMS+1,1>>>(d_cellID,
										  0,
										  NUMBER_OF_ATOMS );
	
	h_initRNG(d_rngStates,
			  NUMBER_OF_ATOMS);
	
	h_generateInitialDist(&cudaPBOres,
						  d_vel,
						  d_acc,
						  d_atomID,
						  NUMBER_OF_ATOMS,
						  d_rngStates);
	
	// Camera
	glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3.0f);
	glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
	
	GLfloat deltaTime = 0.0f;	// Time between current frame and last frame
	GLfloat lastFrame = 0.0f;  	// Time of last frame
	
	int h_numberOfCollisions = 0;
	int h_totalCollisions = 0;
	int h_atomcount = 0;
	
    // Game loop
//    while (!glfwWindowShouldClose(window))
	for (int i=0; i < 1; i++)
    {
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();
		Do_Movement(deltaTime);
		
		h_moveParticles(&cudaPBOres,
						d_vel,
						d_acc,
//						1.e-4*deltaTime,
						1.e-6,
						NUMBER_OF_ATOMS);
		cudaDeviceSynchronize();

		double3 *d_pos = mapCUDAVBOd3(&cudaPBOres);
		
		std::cout << "Colliding Atoms" << std::endl;

		collideAtoms<<<1,1>>>(d_rngStates,
							  d_pos,
							  d_vel,
							  make_double3( -0.01, -0.01, -0.01 ),
							  make_double3(  0.02,  0.02,  0.02 ),
//							  1.e-4*deltaTime,
							  30000.*1.e-6,
							  d_sigvrmax,
							  d_cellStartEnd,
							  d_cellID,
							  d_atomID,
							  d_numberOfCollisions,
							  1,
							  NUMBER_OF_ATOMS,
							  1e6 / NUMBER_OF_ATOMS,
							  d_atomcount,
							  0);
		cudaDeviceSynchronize();
		cudaMemcpy(&h_numberOfCollisions,
				   d_numberOfCollisions,
				   1*sizeof(int),
				   cudaMemcpyDeviceToHost );
		h_totalCollisions = h_totalCollisions + h_numberOfCollisions;
		printf("Total Collisions = %i, Number of Collisions = %i, Collision Rate = %f, <tau> = %f, dt = %f\n\n", h_totalCollisions, h_numberOfCollisions, (float) h_numberOfCollisions / (30000.*1.e-6) / NUMBER_OF_ATOMS, (float) h_totalCollisions / ((i+1)*30000.*1.e-6) / NUMBER_OF_ATOMS * 1e6 / NUMBER_OF_ATOMS * 2.,30000.*1.e-6 );
		h_numberOfCollisions = 0;
		cudaMemcpy(d_numberOfCollisions,
				   &h_numberOfCollisions,
				   1*sizeof(int),
				   cudaMemcpyHostToDevice );
		cudaDeviceSynchronize();
		// Countiing atoms //
		cudaMemcpy(&h_atomcount,
				   d_atomcount,
				   1*sizeof(int),
				   cudaMemcpyDeviceToHost );
		printf("We counted %i atoms, there should be %i\n\n", h_atomcount, NUMBER_OF_ATOMS );
		h_atomcount = 0;
		cudaMemcpy(d_atomcount,
				   &h_atomcount,
				   1*sizeof(int),
				   cudaMemcpyHostToDevice );
		cudaDeviceSynchronize();
		// Finished //
		
		unmapCUDAVBO(&cudaPBOres);
		
		h_setParticleColour(d_vel,
							&cudaCBOres,
							20.e-6,
							NUMBER_OF_ATOMS);
		cudaDeviceSynchronize();
		
		renderParticles(&VAO,
						miShader);
		
        // Swap the screen buffers
        glfwSwapBuffers(window);
		double Ek = 0.;
		double Ep = 0.;
		double T  = 0.;
//		double Ek = calculateKineticEnergy(d_vel,
//										   NUMBER_OF_ATOMS);
//		double Ep = calculatePotentialEnergy(&cudaPBOres,
//											 NUMBER_OF_ATOMS);
//		
//		double T = calculateTemperature(Ek,
//										NUMBER_OF_ATOMS);
		
		frameCount++;
		computeFPS(window,
				   NUMBER_OF_ATOMS,
				   T,
				   (Ep + Ek) / NUMBER_OF_ATOMS,
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
	cudaFree( d_sigvrmax );
	cudaFree( d_atomID );
	cudaFree( d_cellID );
	cudaFree( d_numberOfCollisions );
	cudaFree( d_cellStartEnd );
	cudaFree( d_atomcount );
	
	cudaDeviceReset();
	
    return 0;
}
