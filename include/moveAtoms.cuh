//
//  moveAtoms.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__moveAtoms__
#define __nestedDSMC__moveAtoms__

#include <stdio.h>
#include <iostream>
#include <cuda.h>

#include "magneticField.cuh"

void h_moveParticles(struct cudaGraphicsResource **vbo_resource,
					 double3 *d_vel,
					 double3 *d_acc,
					 double time,
					 int numberOfAtoms);

__global__ void d_moveParticles(double3 *pos,
								double3 *vel,
								double3 *acc,
								double dt,
								int numberOfAtoms);

__device__ void velocityVerletUpdate(double3 *pos,
									 double3 *vel,
									 double3 *acc,
									 double dt);

__device__ void symplecticEulerUpdate(double3 *pos,
									  double3 *vel,
									  double3 *acc,
									  double dt);

__device__ double3 updateVel(double3 vel,
							 double3 acc,
							 double dt);

__device__ double3 updatePos(double3 pos,
							 double3 vel,
							 double dt);

__device__ double3 updateAcc(double3 pos);

#endif /* defined(__nestedDSMC__moveAtoms__) */