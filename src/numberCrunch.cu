//
//  numberCrunch.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 09/04/2015.
//
//

#include "numberCrunch.cuh"
#include "magneticField.cuh"

// Function prototypes

#define cudaCalloc(A, B, C) \
do { \
cudaError_t __cudaCalloc_err = cudaMalloc(A, (B)*C); \
if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, (B)*C); \
} while (0)

__host__ double calculateKineticEnergy(double3 *d_vel,
									   int numberOfAtoms)
{
	double *d_vel2;
	cudaCalloc((void **)&d_vel2,
			   numberOfAtoms,
			   sizeof(double));
	
	h_dot_prod(d_vel,
			   d_vel2,
			   numberOfAtoms);
	
	thrust::device_ptr<double> th_vel2 = thrust::device_pointer_cast( d_vel2 );
	
	double Ek = 0.5 * h_mRb * thrust::reduce(th_vel2,
											 th_vel2 + numberOfAtoms,
											 0.);
	
	cudaFree( d_vel2 );
	
	return Ek;
}

__host__ double calculatePotentialEnergy(struct cudaGraphicsResource **cudaPBOres,
										 int numberOfAtoms)
{
	double *d_absB;
	cudaCalloc((void **)&d_absB,
			   numberOfAtoms,
			   sizeof(double));
	
	double3* d_pos = mapCUDAVBOd3(cudaPBOres);
	
	h_absB(d_pos,
		   d_absB,
		   numberOfAtoms);
	
	thrust::device_ptr<double> th_absB = thrust::device_pointer_cast( d_absB );
	
	double Ep = 0.5 * h_gs * h_muB * thrust::reduce(th_absB,
													th_absB + numberOfAtoms,
													0.);
	
	cudaFree( d_absB );
	unmapCUDAVBO(cudaPBOres);
	
	return Ep;
}

__host__ double calculateTemperature(double Ek,
									 int numberOfAtoms)
{
	return 2. / 3. * Ek / h_kB / numberOfAtoms;
}

void h_dot_prod(double3 *d_in,
				double  *d_out,
				int		 numberOfAtoms)
{
	int blockSize;
	int gridSize;
	
#ifdef CUDA7
	int minGridSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize,
									   &blockSize,
									   (const void *) d_dot_prod,
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
	
	d_dot_prod<<<gridSize,blockSize>>>(d_in,
									   d_out,
									   numberOfAtoms);
	
	return;
}

void h_absB(double3 *d_pos,
			double  *d_absB,
			int		 numberOfAtoms)
{
	int blockSize;
	int gridSize;
	
#ifdef CUDA7
	int minGridSize;
	
	cudaOccupancyMaxPotentialBlockSize(&minGridSize,
									   &blockSize,
									   (const void *) d_getAbsB,
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
	
	d_getAbsB<<<gridSize,blockSize>>>(d_pos,
									  d_absB,
									  numberOfAtoms);
	
	return;
}

__global__ void d_dot_prod(double3 *in,
						   double  *out,
						   int      numberOfAtoms)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		out[atom] = dot( in[atom], in[atom] );
	}
	
	return;
}

__global__ void d_getAbsB(double3 *pos,
						  double  *d_absB,
						  int      numberOfAtoms)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		d_absB[atom] = absB( pos[atom] );
	}
	
	return;
}