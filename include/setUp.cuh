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

#include "moveAtoms.cuh"
#include "openGLHelpers.hpp"

#define NUM_THREADS 256

void h_initRNG(curandState_t *d_rngStates,
			   int sizeOfRNG);
void h_generateInitialDist(struct cudaGraphicsResource **cudaPBOres,
						   double3 *d_vel,
						   double3 *d_acc,
						   int     *d_atomID,
						   int      numberOfAtoms,
						   curandState_t *d_rngStates);

__global__ void d_initRNG(curandState_t *rngState,
						  int numberOfAtoms);
__global__ void d_generateInitialDist(double3 *pos,
									  double3 *vel,
									  double3 *acc,
									  int     *atomID,
									  int      numberOfAtoms,
									  curandState_t *rngState);

__device__ double3 getThermalPosition(double Temp,
									  curandState_t *rngState);

__device__ double3 getThermalVelocity(double Temp,
									  curandState_t *rngState);

__device__ double3 getGaussianPoint(double mean,
									double std,
									curandState_t *rngState);

#endif /* defined(__nestedDSMC__setUp__) */