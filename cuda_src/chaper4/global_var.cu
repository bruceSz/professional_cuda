#include "../common.h"

#include <cuda_runtime.h>
#include <cstdio>

__device__ float devData;

__global__ void checkGD() {
    printf("Device: the value of gb is %f\n", devData);
    devData += 2.0f;
}


int main(void) {
    float val = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &val, sizeof(float)));
    cout << "Host: copied " << val <<
        " to gd of device: " << std::endl;

    checkGD<<<1,1>>>();
    CHECK(cudaMemcpyFromSymbol(&val, devData, sizeof(float)));
    cout << "Host copied: " << val 
        << " from gd of device" << endl;
    
    CHECK(cudaDeviceReset());
}