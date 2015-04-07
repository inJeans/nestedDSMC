//
//  magneticField.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "magneticField.cuh"

#include "vectorMath.cuh"

#include "declareDeviceConstants.cuh"
#include "declareDeviceParameters.cuh"

__device__ double3 B(double3 pos)
{
	double3 B = make_double3( 0., 0., 0. );
	
	/////////////////////////////////////////
	// Make changes to magnetic field here //
	/////////////////////////////////////////
	
	B.x =  0.5 * d_dBdz * pos.x;
	B.y =  0.5 * d_dBdz * pos.y;
	B.z = -1.0 * d_dBdz * pos.z;
	
	return B;
}

__device__ double3 dabsB(double3 pos)
{
	double3 dabsB = make_double3( 0., 0., 0. );
	
	/////////////////////////////////////////
	// Make changes to magnetic field      //
	// derivative here                     //
	/////////////////////////////////////////
	
	dabsB.x = 0.5 * d_dBdz * pos.x * rsqrt( pos.x*pos.x + pos.y*pos.y + 4.*pos.z*pos.z );
	dabsB.y = 0.5 * d_dBdz * pos.y * rsqrt( pos.x*pos.x + pos.y*pos.y + 4.*pos.z*pos.z );
	dabsB.z = 2.0 * d_dBdz * pos.z * rsqrt( pos.x*pos.x + pos.y*pos.y + 4.*pos.z*pos.z );
	
	return dabsB;
}

__device__ double absB(double3 pos)
{
	return sqrt( dot( B(pos), B(pos) ) );
}