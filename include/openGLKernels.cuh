//
//  openGLkernels.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__openGLkernels__
#define __nestedDSMC__openGLkernels__

#include <math.h>

void h_setParticleColour(double3 *d_vel,
						 struct   cudaGraphicsResource **cudaCBOres,
						 double   T,
						 int      numberOfAtoms);

__global__ void d_setParticleColour(double3 *vel,
									float4  *col,
									double   T,
									int      numberOfAtoms);

#endif /* defined(__nestedDSMC__openGLkernels__) */