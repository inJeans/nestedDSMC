//
//  setUp.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "setUp.cuh"

#include "vectorMath.cuh"

#include "declareHostConstants.h"
#include "declareHostParameters.h"
#include "defineDeviceConstants.cuh"
#include "defineDeviceParameters.cuh"

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
	gridSize = (sizeOfRNG + blockSize - 1) / blockSize;
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
	fprintf(stderr, "h_initRNG Error: %s\n", cudaGetErrorString( cudaGetLastError( ) ) );
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

void h_generateInitialDist(struct cudaGraphicsResource **cudaPBOres,
						   double3 *d_vel,
						   double3 *d_acc,
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
	
	// Map OpenGL buffer object for writing from CUDA
	double3 *d_pos = mapCUDAVBOd3(cudaPBOres);
	
	d_generateInitialDist<<<gridSize,blockSize>>>(d_pos,
												  d_vel,
												  d_acc,
												  numberOfAtoms,
												  d_rngStates);
	
	//Unmap buffer object
	unmapCUDAVBO(cudaPBOres);
	
	return;
}

// Kernel to generate the initial distribution
__global__ void d_generateInitialDist(double3 *pos,
                                      double3 *vel,
									  double3 *acc,
                                      int      numberOfAtoms,
                                      curandState_t *rngState)
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        /* Copy state to local memory for efficiency */
        curandState_t l_rngState = rngState[atom];
		
		pos[atom] = getThermalPosition(20.e-6,
									   &l_rngState);
		
		vel[atom] = getThermalVelocity(20.e-6,
									   &l_rngState);
		
		acc[atom] = updateAcc(pos[atom]);
		
        // Copy state back to global memory
        rngState[atom] = l_rngState;
		
    }
	
    return;
}

__device__ double3 getThermalPosition(double Temp,
									  curandState_t *rngState)
{
	double3 r   = make_double3( 0., 0., 0. );
	double3 pos = make_double3( 0., 0., 0. );
	
	bool noAtomSelected = true;
	while (noAtomSelected) {
		double2 r1 = curand_normal2_double ( &rngState[0] );
		double  r2 = curand_normal_double  ( &rngState[0] );
		
		double thermalWidth = 12. * d_kB * Temp / ( d_gs * d_muB * d_dBdz );
		
		double3 r = make_double3( r1.x, r1.y, r2 ) * 4. * thermalWidth;
		
		double U = -0.5 * d_gs * d_muB * absB( r );;
		
		double Pr = exp( U / d_kB / Temp );
		
		if ( curand_uniform_double ( &rngState[0] ) < Pr) {
			pos = r;
			noAtomSelected = false;
		}
	}
	
	return pos;
}

__device__ double3 getThermalVelocity(double Temp,
									  curandState_t *rngState)
{
	double3 vel = make_double3( 0., 0., 0. );
	
	double V = sqrt( d_kB*Temp/d_mRb );
	
	vel = getGaussianPoint(0.,
						   V,
						   &rngState[0]);
	
	return vel;
}

__device__ double3 getGaussianPoint(double mean,
									double std,
									curandState_t *rngState)
{
	double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double  r2 = curand_normal_double  ( &rngState[0] ) * std + mean;
 
	double3 point = make_double3( r1.x, r1.y, r2 );
	
	return point;
}
