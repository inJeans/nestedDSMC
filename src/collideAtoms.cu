//
//  collideAtoms.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 11/04/2015.
//
//

#include "collideAtoms.cuh"

//#include <thrust/device_vector.h>
//#include <thrust/sort.h>
#include <cub/cub.cuh>

#include "vectorMath.cuh"

#include "declareDeviceConstants.cuh"
#include "declareDeviceParameters.cuh"

__constant__ int3 cellsPerAxis = { 2, 2, 2 };
#define newNumberOfSubCells 8

static __inline__ __device__ int3 double3Toint3_rd( double3 a )
{
	return make_int3( __double2int_rd(a.x), __double2int_rd(a.y), __double2int_rd(a.z) );
}

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
							 int level)
{
	for (int subCell = blockIdx.x * blockDim.x + threadIdx.x;
		 subCell < numberOfSubCells;
		 subCell += blockDim.x * gridDim.x)
	{
		int2 l_cellStartEnd = cellStartEnd[subCell];
		int  numberOfAtoms  = getNumberOfAtoms(l_cellStartEnd);
		
		double3 subCellWidth = cellWidth / cellsPerAxis;
		
		double3 subCellMin = cellMin + cellWidth * getSubcellIndex(subCell,
																	  cellsPerAxis);
		double3 subCellMax = subCellMin + cellWidth;
		
		if (numberOfAtoms > d_Nth)
		{
			int blockSize = 1024;
			int gridSize = numberOfAtoms / blockSize + 1;
			
			setCellID<<<gridSize,blockSize>>>(pos,
											  &cellID[l_cellStartEnd.x],
											  &atomID[l_cellStartEnd.x],
											  subCellMin,
											  subCellWidth,
											  cellsPerAxis,
											  numberOfAtoms);
//			__syncthreads();
//			cudaDeviceSynchronize();
//			__syncthreads();
			
			sortArrays(&cellID[l_cellStartEnd.x],
					   &atomID[l_cellStartEnd.x],
					   numberOfAtoms);
//			__syncthreads();
//			cudaDeviceSynchronize();
//			__syncthreads();
			
			int2 *subCellStartEnd;
			cudaMalloc((void **)&subCellStartEnd,
					   newNumberOfSubCells * sizeof(int2));
			
			for (int i=0; i<newNumberOfSubCells; i++) {
				subCellStartEnd[i] = make_int2( -1, -1 );
			}
			printf("**Hi - find cell start and end - %i**\n", level+1);
			
//			getCellStartEnd<<<gridSize,blockSize>>>(&cellID[l_cellStartEnd.x],
			getCellStartEnd<<<1,1>>>(&cellID[l_cellStartEnd.x],
													subCellStartEnd,
													numberOfAtoms,
													l_cellStartEnd);
			cudaDeviceSynchronize();
			
			for (int i=0; i<8; i++)
			{
				if (cellStartEnd[i].y <  cellStartEnd[i].x)
				{
					printf("level%i - cell%i - cellStartEnd = {%i, %i}\n",level+1, i,cellStartEnd[i].x,cellStartEnd[i].y);
					printf("cellID[%i] = %i, cellID[%i] = %i, cellID[%i] = %i\n", cellStartEnd[i].x-1, cellID[cellStartEnd[i].x-1],
						                                                          cellStartEnd[i].x  , cellID[cellStartEnd[i].x  ],
						                                                          cellStartEnd[i].x+1, cellID[cellStartEnd[i].x+1]);
				}
			}
//			__syncthreads();
//			cudaDeviceSynchronize();
//			__syncthreads();
			
			collideAtoms<<<1,newNumberOfSubCells>>>(rngState,
													pos,
													vel,
													subCellMin,
													subCellWidth,
													dt,
													sigvrmax,
													subCellStartEnd,
													&cellID[l_cellStartEnd.x],
													&atomID[l_cellStartEnd.x],
													numberOfCollisions,
													newNumberOfSubCells,
													numberOfAtoms,
													alpha,
													atomcount,
													level+1);
			
			cudaFree(subCellStartEnd);
		}
		else
		{
			if (numberOfAtoms > 1) {
				double sigma = 8.*d_pi*d_a*d_a;
				double subCellVolume = subCellWidth.x * subCellWidth.y * subCellWidth.z;
				double Mc = 0.5 * (numberOfAtoms - 1) * numberOfAtoms;
				double lambda = ceil( Mc * alpha * dt * sigvrmax[0] / subCellVolume ) / Mc;
				double Ncol = Mc*lambda;
				
				curandState_t l_rngState = rngState[subCell];
				
				double ProbCol = 0.;
				int collisionCount = 0;
				
				for (int c=0; c<Ncol; c++) {
					int2 collidingAtoms = chooseCollidingAtoms(&l_rngState,
															   numberOfAtoms);
					
					double normvrel = length( calculateRelativeVelocity(vel,
																		collidingAtoms ) );
					
					// Check if this is the more probable than current most probable.
					if (normvrel*sigma > sigvrmax[0]) {
						sigvrmax[0] = normvrel * sigma;
					}
					
					ProbCol = alpha * dt / subCellVolume * normvrel * sigma / lambda;
					
					// Collide with the collision probability.
					if ( ProbCol > curand_uniform_double ( &l_rngState ) ) {
						collisionCount++;
					}
				}
				
				atomicAdd(numberOfCollisions, collisionCount);
				atomicAdd(atomcount, numberOfAtoms);
				rngState[subCell] = l_rngState;
				
				printf("Natoms = %i, Ncol = %f, ProbCol = %f, colCount = %i, numberOfCollisions = %i\n", numberOfAtoms, Ncol, ProbCol, collisionCount, numberOfCollisions[0]);
			}
			else
			{
				printf("Natoms = %i\n", numberOfAtoms);
				atomicAdd(atomcount, numberOfAtoms);
			}
			
		}
//		__syncthreads();
//		cudaDeviceSynchronize();
//		__syncthreads();
		
	}
	
	return;
}

#pragma mark - Indexing

__global__ void setCellID(double3 *pos,
						  int     *cellID,
						  int     *atomID,
						  double3  cellMin,
						  double3  cellWidth,
						  int3     cellsPerAxis,
						  int numberOfAtoms)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		double3 l_pos = pos[atomID[atom]];
		int3 cellIndex = getCellIndex(l_pos,
									  cellMin,
									  cellWidth);
		
		cellID[atom] = getCellID(cellIndex,
								 cellsPerAxis);
	}
	return;
}

__device__ int3 getCellIndex(double3 pos,
							 double3 cellMin,
							 double3 cellWidth)
{
	return double3Toint3_rd ( (pos - cellMin) / cellWidth );
}

__device__ int getCellID(int3 index,
						 int3 cellsPerAxis)
{
	int cellID = 0;
	
	if (index.x > -1 && index.x < cellsPerAxis.x &&
		index.y > -1 && index.y < cellsPerAxis.y &&
		index.z > -1 && index.z < cellsPerAxis.z)
	{
		cellID = index.z*cellsPerAxis.x*cellsPerAxis.y + index.y*cellsPerAxis.x + index.x;
	}
	else
	{
		cellID = cellsPerAxis.x * cellsPerAxis.y * cellsPerAxis.z;
	}
	
	return cellID;
}

__device__ int3 getSubcellIndex(int   cellID,
								int3  cellsPerAxis)
{
	int3 index = { 0, 0, 0 };
	
	index.z = cellID / (cellsPerAxis.x*cellsPerAxis.y);
	index.y = (cellID - index.z*(cellsPerAxis.x*cellsPerAxis.y)) / cellsPerAxis.x;
	index.x = cellID - index.z*cellsPerAxis.x*cellsPerAxis.y - index.y*cellsPerAxis.x;
								
	return index;
}

#pragma mark - Sorting

__device__ void sortArrays(int *cellID,
						   int *atomID,
						   int  numberOfAtoms)
{
	
	int *cellID_alt_buff;
	int *atomID_alt_buff;
	
	cudaMalloc(&cellID_alt_buff,
			   numberOfAtoms * sizeof(int));
	cudaMalloc(&atomID_alt_buff,
			   numberOfAtoms * sizeof(int));
	
	cub::DoubleBuffer<int> d_keys(cellID,
								  cellID_alt_buff);
	cub::DoubleBuffer<int> d_values(atomID,
									atomID_alt_buff);
	
	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortPairs(d_temp_storage,
									temp_storage_bytes,
									d_keys,
									d_values,
									numberOfAtoms);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage,
			   temp_storage_bytes);
	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(d_temp_storage,
									temp_storage_bytes,
									d_keys,
									d_values,
									numberOfAtoms);
	
	cudaFree(cellID_alt_buff);
	cudaFree(atomID_alt_buff);
	cudaFree(d_temp_storage);
	
	return;
}

#pragma mark - Counting

__global__ void getCellStartEnd(int  *cellID,
								int2 *cellStartEnd,
								int   numberOfAtoms,
								int2 l_cellStartEnd)
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		// Find the beginning of the cell
		if (atom == 0) {
			cellStartEnd[cellID[atom]].x = 0;
		}
		else if (cellID[atom] != cellID[atom-1]) {
			cellStartEnd[cellID[atom]].x = atom;
		}
		
		// Find the end of the cell
		if (atom == numberOfAtoms - 1) {
			cellStartEnd[cellID[atom]].y = atom;
			printf("end atom-%i, cell-%i\n", atom, cellID[atom]);
		}
		else if (cellID[atom] != cellID[atom+1]) {
			cellStartEnd[cellID[atom]].y = atom;
		}
	}
	
	return;
}

__device__ int getNumberOfAtoms(int2 cellStartEnd)
{
	int numberOfAtoms = 0;
	
	numberOfAtoms  = cellStartEnd.y - cellStartEnd.x + 1;
	
	if (numberOfAtoms < 0) {
		numberOfAtoms = 0;
	}
	
	return numberOfAtoms;
}

#pragma mark - Colliding

__device__ int2 chooseCollidingAtoms(curandState_t *rngState,
									 int numberOfAtoms) 
{
	int2 collidingAtoms = { 0, 0 };
	
	if (numberOfAtoms == 2)
	{
		collidingAtoms.x = 0;
		collidingAtoms.y = 1;
	}
	else
	{
		// Randomly choose particles in this cell to collide.
		while (collidingAtoms.x == collidingAtoms.y) {
			collidingAtoms = double2Toint2_rd(make_double2(curand_uniform_double ( rngState ),
														   curand_uniform_double ( rngState ) ) * (numberOfAtoms-1) );
		}
	}
	
	return collidingAtoms;
}

__device__ double3 calculateRelativeVelocity(double3 *vel,
											 int2 collidingAtoms )
{
	double3 relVel = { 0., 0., 0. };
	
	relVel = vel[collidingAtoms.x] - vel[collidingAtoms.y];
	
	return relVel;
	
}