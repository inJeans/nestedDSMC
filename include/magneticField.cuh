//
//  magneticField.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#ifndef __nestedDSMC__magneticField__
#define __nestedDSMC__magneticField__

#include <stdio.h>
#include <iostream>
#include <math.h>

__device__ double3 B(double3 pos);

__device__ double3 dabsB(double3 pos);

__device__ double absB(double3 pos);

#endif /* defined(__nestedDSMC__magneticField__) */