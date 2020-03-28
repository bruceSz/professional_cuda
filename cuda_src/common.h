#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


void getCudaInfo() {
    int dev =0;
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, dev);
    cudaSetDevice(dev);

    std::cout << "Using device : " << dprop.name << std::endl;
    

}