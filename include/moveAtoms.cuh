//
//  moveAtoms.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__moveAtoms__
#define __nestedDSMC__moveAtoms__

#include <stdio.h>
#include <cuda.h>

__global__ void moveParticles( float3 *pos, double time, int numberOfAtoms );

#endif /* defined(__nestedDSMC__moveAtoms__) */