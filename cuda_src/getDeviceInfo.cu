#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

using namespace std;


__global__ void checkIndex(void) {

//    printf("- thread idx is : ");
    printf( "thread idx:  %d, %d, %d\n" , threadIdx.x , threadIdx.y , threadIdx.z );
    printf( "block idx: %d ,  %d, %d\n", blockIdx.x , blockIdx.y , blockIdx.z );
    printf ("block dim: %d , %d, %d\n" ,blockDim.x , blockDim.y , blockDim.z );
    printf(  "grid dim: %d , %d, %d\n", gridDim.x , gridDim.y , gridDim.z); 
    printf("-");
}

int main() {
    int  n = 10;
    dim3 block(3);
    dim3 grid((n + block.x -1)/block.x);

    cout << " grid x: " << grid.x << " grid.y: " << grid.y << " grid.z : " << grid.z << std::endl;
    cout << " block x: " << block.x << " block.y : " << block.y<< " block.z: " << block.z << std::endl;

    checkIndex<<<grid, block>>>();
    cudaDeviceSynchronize();

}

int main1(int argc, char** argv) {
    std::cout << " Starting." << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cout << "There is no device available." << std::endl;  
    } else {
        cout << "Detected " << deviceCount << " cuda capable device." << endl;
    }

    int dev = 0;

    cudaDeviceProp deviceProp;
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "Deivce id: " << dev << " " << deviceProp.name << endl;

    int dversion, runtimeVersion;
    cudaDriverGetVersion(&dversion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << "CUDA driver version : " << dversion << " runtime version: " << runtimeVersion << endl;
    cout << "Total mem: " << deviceProp.totalGlobalMem/(1024*1024*1024) << endl;

    cudaDeviceReset();
    cout << "After device reset." << std::endl;

    //int dev2 = 2;
    //cudaSetDevice(dev2);
    //cudaGetDeviceProperties(&deviceProp, dev2);
    //cout << "Deivce id: " << dev2 << " " << deviceProp.name << endl;

    return 1;
}
