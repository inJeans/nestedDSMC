//
//  numberCrunch.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 09/04/2015.
//
//

#ifndef __nestedDSMC__numberCrunch__
#define __nestedDSMC__numberCrunch__

#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>

#include "vectorMath.cuh"

#include "declareHostConstants.h"
#include "declareHostParameters.h"

#include "openGLHelpers.hpp"

__host__ double calculateKineticEnergy(double3 *d_vel,
									   int numberOfAtoms);

__host__ double calculatePotentialEnergy(struct cudaGraphicsResource **cudaPBOres,
										 int numberOfAtoms);

__host__ double calculateTemperature(double Ek,
									 int numberOfAtoms);

void h_dot_prod(double3 *d_in,
				double  *d_out,
				int		 numberOfAtoms);

void h_absB(double3 *d_pos,
			double  *d_absB,
			int		 numberOfAtoms);

__global__ void d_dot_prod(double3 *in,
						   double  *out,
						   int      numberOfAtoms);

__global__ void d_getAbsB(double3 *pos,
						  double  *absB,
						  int      numberOfAtoms);

#endif /* defined(__nestedDSMC__numberCrunch__) */