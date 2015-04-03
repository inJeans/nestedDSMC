//
//  setUp.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__setUp__
#define __nestedDSMC__setUp__

#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#include "systemParameters.h"
#include "vectorMath.cuh"

#define NUM_THREADS 256

__device__ double d_pi = 6.2831853072;

void h_initRNG(curandState_t *d_rngStates,
			   int sizeOfRNG);
void h_generateInitialDist(struct cudaGraphicsResource **cudaVBOres,
						   double3 *d_vel,
						   int      numberOfAtoms,
						   curandState_t *d_rngStates);

__global__ void d_initRNG(curandState_t *rngState,
						  int numberOfAtoms);
__global__ void d_generateInitialDist(double3 *pos,
									  double3 *vel,
									  int      numberOfAtoms,
									  curandState_t *rngState);

__device__ double3 createPointOnCircle(int atom,
									   int numberOfAtoms);

#endif /* defined(__nestedDSMC__setUp__) */