#include "setUp.cuh"

void h_initRNG(curandState_t *d_rngStates,
			   int sizeOfRNG)
{
	int blockSize;
	int gridSize;
	
#ifdef CUDA7
	int minGridSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize,
									   &blockSize,
									   (const void *) d_initRNG,
									   0,
									   sizeOfRNG);
	gridSize = (NUMBER_OF_ATOMS + blockSize - 1) / blockSize;
#else
	int device;
	cudaGetDevice ( &device );
	int numSMs;
	cudaDeviceGetAttribute(&numSMs,
						   cudaDevAttrMultiProcessorCount,
						   device);
	
	gridSize = 256*numSMs;
	blockSize = NUM_THREADS;
#endif
	
	d_initRNG<<<gridSize,blockSize>>>(d_rngStates,
									  sizeOfRNG);
	
	return;
}

__global__ void d_initRNG(curandState_t *rngState,
						  int numberOfAtoms)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		// Each thread gets the same seed, a different sequence
		// number and no offset
		curand_init( 1234, atom, 0, &rngState[atom] );
	}
	
	return;
}

void h_generateInitialDist(struct cudaGraphicsResource **cudaVBOres,
						   double3 *d_vel,
						   int      numberOfAtoms,
						   curandState_t *d_rngStates)
{
	int blockSize;
	int gridSize;
	
#ifdef CUDA7
	int minGridSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize,
									   &blockSize,
									   (const void *) d_generateInitialDist,
									   0,
									   numberOfAtoms );
	gridSize = (NUMBER_OF_ATOMS + blockSize - 1) / blockSize;
#else
	int device;
	cudaGetDevice ( &device );
	int numSMs;
	cudaDeviceGetAttribute(&numSMs,
						   cudaDevAttrMultiProcessorCount,
						   device);
	
	gridSize = 256*numSMs;
	blockSize = NUM_THREADS;
#endif
	
	// Map OpenGL buffer object for writing from CUDA
	double3 *d_pos;
	cudaGraphicsMapResources(1,
							 cudaVBOres,
							 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_pos,
										 &num_bytes,
										 *cudaVBOres);
	
	d_generateInitialDist<<<gridSize,blockSize>>>(d_pos,
												  d_vel,
												  numberOfAtoms,
												  d_rngStates);
	
	//Unmap buffer object
	cudaGraphicsUnmapResources(1,
							   cudaVBOres,
							   0);
	
	return;
}

// Kernel to generate the initial distribution
__global__ void d_generateInitialDist(double3 *pos,
                                      double3 *vel,
                                      int      numberOfAtoms,
                                      curandState_t *rngState)
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        /* Copy state to local memory for efficiency */
        curandState_t localrngState = rngState[atom];
		
		pos[atom] = createPointOnCircle(atom,
										numberOfAtoms);
		
        vel[atom] = make_double3( 0., 0., 0. );
        
        // Copy state back to global memory
        rngState[atom] = localrngState;
		
    }
	
    return;
}

__device__ double3 createPointOnCircle(int atom,
									   int numberOfAtoms)
{
	double3 pos = make_double3(0.,
							   0.,
							   0. );
	
	pos.x = atom*cos( 2. * d_pi * atom / numberOfAtoms ) / numberOfAtoms;
	pos.y = atom*sin( 2. * d_pi * atom / numberOfAtoms ) / numberOfAtoms;
	pos.z = 0.;
	
	return pos;
}
