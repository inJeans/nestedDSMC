//
//  collideAtoms.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 11/04/2015.
//
//

#ifndef __nestedDSMC__collideAtoms__
#define __nestedDSMC__collideAtoms__

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>

#include "cudaHelpers.cuh"

__global__ void collideAtoms(curandState_t *rngState,
							 double3 *pos,
							 double3 *vel,
							 double3  cellMin,
							 double3  cellWidth,
							 double   dt,
							 double  *sigvrmax,
							 int2    *cellStartEnd,
							 int     *cellID,
							 int     *atomID,
							 int     *numberOfCollisions,
							 int      numberOfSubCells,
							 int      num_atoms,
							 int      alpha,
							 int *atomcount,
							 int level);

__global__ void setCellID(double3 *pos,
						  int     *cellID,
						  int     *atomID,
						  double3  cellMin,
						  double3  cellMax,
						  int3     cellsPerAxis,
						  int numberOfAtoms);

__device__ int3 getCellIndex(double3 pos,
							 double3 cellMin,
							 double3 cellMax);

__device__ int getCellID(int3 index,
						 int3 cellsPerAxis);

__device__ int3 getSubcellIndex(int   cellID,
								int3  cellsPerAxis);

__device__ void sortArrays(int *d_cellID,
						   int *d_atomID,
						   int numberOfAtoms);

__global__ void getCellStartEnd(int  *cellID,
								int2 *cellStartEnd,
								int   numberOfAtoms,
								int2  l_cellStartEnd);

__device__ int getNumberOfAtoms(int2 cellStartEnd);

__device__ int2 chooseCollidingAtoms(curandState_t *rngState,
									 int numberOfAtoms);

__device__ double3 calculateRelativeVelocity(double3 *vel,
											 int2 collidingAtoms );

#endif /* defined(__nestedDSMC__collideAtoms__) */