#include <sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}



#define CHECK(call) \
{\
    const cudaError_t err = call;\
    if( err != cudaSuccess) \
    {\
        fprintf(stderr, "Error: %s:%d ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", err, cudaGetErrorString(err));\
    }\
}


cudaDeviceProp getCudaInfo() {
    int dev =0;
    cudaDeviceProp dprop;
    cudaGetDeviceProperties(&dprop, dev);
    cudaSetDevice(dev);

    std::cout << "Using device : " << dprop.name << std::endl;
    cout << "Device canMapHostMemory : " << dprop.canMapHostMemory << endl;
    return dprop;

}