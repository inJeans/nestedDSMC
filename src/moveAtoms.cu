//
//  moveAtoms.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "moveAtoms.cuh"

#include "vectorMath.cuh"

#include "declareDeviceConstants.cuh"
#include "declareDeviceParameters.cuh"

void h_moveParticles(struct cudaGraphicsResource **cudaVBOres,
					 double3 *d_vel,
					 double3 *d_acc,
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
	
	d_moveParticles<<< 1, 4 >>>(d_pos,
								d_vel,
								d_acc,
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
								double3 *acc,
								double dt,
								int numberOfAtoms)
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
//    {
//		double3 acc = -1.0 * pos[atom];
//		
//		vel[atom] = vel[atom] + acc * dt;
//		pos[atom] = pos[atom] + vel[atom] * dt;
//    }
	{
		double3 l_pos = pos[atom];
		double3 l_vel = vel[atom];
		double3 l_acc = acc[atom];
		
		//        for (int i=0; i<d_loopsPerCollision; i++) {
		velocityVerletUpdate(&l_pos,
							 &l_vel,
							 &l_acc,
							 dt);
		//        }
		
		pos[atom] = l_pos;
		vel[atom] = l_vel;
		acc[atom] = l_acc;
	}

    return;
	
}

__device__ void velocityVerletUpdate(double3 *pos,
									 double3 *vel,
									 double3 *acc,
									 double dt)
{
	vel[0] = updateVel(vel[0],
					   acc[0],
					   0.5*dt);
	pos[0] = updatePos(pos[0],
					   vel[0],
					   dt);
	acc[0] = updateAcc(pos[0]);
	vel[0] = updateVel(vel[0],
					   acc[0],
					   0.5*dt);
	
	return;
}

__device__ void symplecticEulerUpdate(double3 *pos,
									  double3 *vel,
									  double3 *acc,
									  double dt)
{
	acc[0] = updateAcc(pos[0]);
	vel[0] = updateVel(vel[0],
					   acc[0],
					   dt);
	pos[0] = updatePos(pos[0],
					   vel[0],
					   dt);
}

__device__ double3 updateVel(double3 vel,
							 double3 acc,
							 double dt)
{
	return vel + acc * dt;
}

__device__ double3 updatePos(double3 pos,
							 double3 vel,
							 double dt)
{
	return pos + vel * dt;
}

__device__ double3 updateAcc(double3 pos)
{
	
	return -0.5 * d_gs * d_muB * dabsB(pos) / d_mRb;
}