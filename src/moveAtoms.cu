//
//  moveAtoms.cu
//  nestedDSMC
//
//  Created by Christopher Watkins on 24/03/2015.
//
//

#include "moveAtoms.cuh"

__global__ void moveParticles( float3 *pos, double time, int numberOfAtoms )
{
    printf("Hello\n");
//    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
//         atom < numberOfAtoms;
//         atom += blockDim.x * gridDim.x)
//    {
//        
//        double offset = time;
//        if (time > 1.5) {
//            offset = -1.5 + fmod( time - 1.5, 3. );
//        }
//        
//        pos[atom].x += offset;
//        
//        printf("pos %i = {%f,%f,%f}\n", atom, pos[atom].x, pos[atom].y, pos[atom].z);
//    }
    
    return;
    
}