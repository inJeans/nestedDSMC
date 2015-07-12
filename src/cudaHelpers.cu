//
//  cudaHelpers.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 12/04/2015.
//
//

#include "cudaHelpers.cuh"


#include <stdio.h>

__device__ void d_cudaCalloc(int2 *array,
							 int2  init,
							 int   size)
{
	cudaMalloc((void **)&array,
			   size * sizeof(int2));
	
	for (int i=0; i<size; i++) {
		array[i] = init;
		printf("array[%i] = {%i, %i}\n", i, array[i].x, array[i].y);
	}
	
	return;
}

__global__ void deviceMemset(double *d_array,
							 double  value,
							 int     lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		 element < lengthOfArray;
		 element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}

__global__ void deviceMemset(int2 *d_array,
							 int2 value,
							 int  lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		 element < lengthOfArray;
		 element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}

__global__ void deviceMemset(int *d_array,
							 int value,
							 int lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		 element < lengthOfArray;
		 element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}