//
//  cudaHelpers.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 12/04/2015.
//
//

#ifndef __nestedDSMC__cudaHelpers__
#define __nestedDSMC__cudaHelpers__

__device__ void d_cudaCalloc(int2 *ptr,
							 int2  init,
							 int   size);

__global__ void deviceMemset(double *d_array,
							 double  value,
							 int     lengthOfArray );

__global__ void deviceMemset(int2 *d_array,
							 int2 value,
							 int  lengthOfArray );

__global__ void deviceMemset(int *d_array,
							 int value,
							 int lengthOfArray );

#endif /* defined(__nestedDSMC__cudaHelpers__) */