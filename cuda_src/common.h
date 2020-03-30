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


void checkRes(float* host, float* gpu, const int N) {
    double eps = 1.0E-8;
    for(int i=0;i<N; i++) {
        if(abs(host[i] - gpu[i]) > eps) {
            printf("host %f while gpu %f", host[i], gpu[i]);
            printf(" Array not match at %d.\n\n", i);
            break;
        }
    }
}

void initD(float* ip, const int size) {
    int i;
    for(i=0;i<size;i++)
     {
         ip[i] = (float)(rand() & 0xFF) / 10.0f;
     }
}


void sumArrayOnHost(float* A, float* B, float* C, const int n, int offset) {
    for(int idx = offset, k=0;idx< n;idx++, k++) {
        C[k] = A[idx] + B[idx];
    }
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