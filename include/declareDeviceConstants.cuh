//
//  declareDeviceConstants.cuh
//  nestedDSMC
//
//  Created by Christopher Watkins on 03/04/2015.
//
//

#ifndef __nestedDSMC__declareDeviceConstants__
#define __nestedDSMC__declareDeviceConstants__

extern __constant__ double d_gs;	// Gyromagnetic ratio
extern __constant__ double d_muB;	// Bohr magneton
extern __constant__ double d_mRb;	// Rb87 mass
extern __constant__ double d_pi;	// pi
extern __constant__ double d_a;		// Constant cross-section formula
extern __constant__ double d_kB;	// Boltzmann's Constant
extern __constant__ double d_hbar ;	// hbar

#endif /* defined(__nestedDSMC__declareDeviceConstants__) */