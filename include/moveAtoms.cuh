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

#include "vectorMath.cuh"

void moveParticles(struct cudaGraphicsResource **vbo_resource,
				   double3 *d_vel,
				   double time,
				   int numberOfAtoms);

__global__ void d_moveParticles(double3 *pos,
								double3 *vel,
								double dt,
								int numberOfAtoms);

#endif /* defined(__nestedDSMC__moveAtoms__) */