//
//  openGLKernels.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "openGLKernels.cuh"

#include "vectorMath.cuh"

#include "declareDeviceConstants.cuh"
#include "declareDeviceParameters.cuh"

#include <stdio.h>

void h_setParticleColour(double3 *d_vel,
						 struct   cudaGraphicsResource **cudaCBOres,
						 double   T,
						 int      numberOfAtoms)
{
	// Map OpenGL buffer object for writing from CUDA
	float4 *d_col;
	cudaGraphicsMapResources(1,
							 cudaCBOres,
							 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_col,
										 &num_bytes,
										 *cudaCBOres);
	
	int blockSize;
	int gridSize;
	
#ifdef CUDA7
	int minGridSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize,
									   &blockSize,
									   (const void *) d_setParticleColour,
									   0,
									   numberOfAtoms);
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
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
	
	d_setParticleColour<<<gridSize,blockSize>>>(d_vel,
												d_col,
												T,
												numberOfAtoms);
	
	//Unmap buffer object
	cudaGraphicsUnmapResources(1,
							   cudaCBOres,
							   0);
	
	return;
}

__global__ void d_setParticleColour(double3 *vel,
									float4  *col,
									double   T,
									int      numberOfAtoms)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		float vavg  = sqrt( 8. * d_kB * T / d_mRb / d_pi );
		float alpha = 0.5 * length(vel[atom]) / vavg;
		
		col[atom] = make_float4( alpha, 0.0, 1. - alpha, 1.0 );
		
//		if (atom==0) {
//			printf("col[%i] = {%g,%g,%g,%g}\n",atom,col[atom].x,col[atom].y,col[atom].z,col[atom].w);
//		}
		
	}
	
	return;
}