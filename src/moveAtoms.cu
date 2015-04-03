//
//  moveAtoms.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "moveAtoms.cuh"

void moveParticles(struct cudaGraphicsResource **cudaVBOres,
				   double3 *d_vel,
				   double timeValue,
				   int numberOfAtoms)
{
	// Map OpenGL buffer object for writing from CUDA
	double3 *d_pos;
	cudaGraphicsMapResources(1,
							 cudaVBOres,
							 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_pos,
										 &num_bytes,
										 *cudaVBOres);
//	printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
	
	d_moveParticles<<< 1, 4 >>>(d_pos,
								d_vel,
								timeValue,
								numberOfAtoms);
	
	//Unmap buffer object
	cudaGraphicsUnmapResources(1,
							   cudaVBOres,
							   0);
	
	return;
}

__global__ void d_moveParticles(double3 *pos,
								double3 *vel,
								double dt,
								int numberOfAtoms)
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
		double3 acc = -1.0 * pos[atom];
		
		vel[atom] = vel[atom] + acc * dt;
		pos[atom] = pos[atom] + vel[atom] * dt;
    }
	
    return;
    
}