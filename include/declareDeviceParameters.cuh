//
//  declareDeviceParameters.h
//  nestedDSMC
//
//  Created by Christopher Watkins on 03/04/2015.
//
//

#ifndef __nestedDSMC__declareDeviceParameters__
#define __nestedDSMC__declareDeviceParameters__

extern __constant__ int d_NUMBER_OF_ATOMS;

////////////////////////////////////////
// Trapping Parameters                //
////////////////////////////////////////

extern __device__ double d_dBdz;


#endif /* defined(__nestedDSMC__declareDeviceParameters__) */