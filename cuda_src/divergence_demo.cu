#include <cuda_runtime.h>
#include <iostream>
#include "common.h"

using namespace std;

__global__ void k1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;

    ia = ib = 0.0f;

    // divergence here.
    if(tid %2==0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}



__global__ void k2(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    // in one warp the branch is same 
    // hence there is no divergence in single warp.
    if((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    
    c[tid] = ia + ib;
}


__global__ void k3(float* c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    //printf("warpsize is %d", warpSize);

    bool ipred = (tid %2 ==0);

    // amazingly this is faster k1 and k2. 
    // maybe optimized by compiler
    if(tid%2==0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


__global__ void k4(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia  = ib = 0.0f;

    int itid = tid>>5;
    // number of warp

    // same as k2
    if(itid & 0x01 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void k5(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia  = ib = 0.0f;

    int itid = tid>>5;
    // number of warp
    bool pred = (itid & 0x01 == 0);
    // same as k2
    if(pred) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaDeviceProp dev_prop;


    //cout << " warp size: " << warpSize << std::endl;
    cudaGetDeviceProperties(&dev_prop, dev);

    std::cout << "using device " << dev_prop.name << std::endl; 

    int size = 8192;
    int blocksize = 8192;
    int test_no = 1;

    if(argc > 1) test_no = atoi(argv[1]);
    if(argc > 2) blocksize = atoi(argv[2]);
    if (argc > 3) size = atoi(argv[3]);

    std::cout << "data size: " << size << endl;


    dim3 block(blocksize,1);
    dim3 grid((size + block.x -1)/block.x , 1);


    float* c;


    size_t n_b = size * sizeof(float);

    cudaMalloc((float**)&c, n_b);


    double start, elaps;


    if ( test_no == 1) {
        cout << " test on k1" << std::endl;
        start = seconds();
        k1<<<grid,block>>>(c);
    
        cudaDeviceSynchronize();
        elaps = seconds() - start;
        cout << " k1 finished with ts: " << elaps << std::endl;    
    } else if(test_no ==2 ) {
        cout << " test on k2" << std::endl;
        start = seconds();
        k2<<<grid,block>>>(c);
        cudaDeviceSynchronize();
        elaps = seconds() - start;
        cout << " k2 finished with ts: " << elaps << std::endl;

    } else if(test_no == 3) {
        cout << " test on k3" << std::endl;
        start = seconds();
        k3<<<grid,block>>>(c);
    
        cudaDeviceSynchronize();
        elaps = seconds() - start;
        cout << " k3 finished with ts: " << elaps << std::endl;
    } else if(test_no == 4) {
        cout << " test on k4" << std::endl;
        start = seconds();
        k4<<<grid,block>>>(c);    
        cudaDeviceSynchronize();
        elaps = seconds() - start;
        cout << " k4 finished with ts: " << elaps << std::endl;

    } else if(test_no == 5) {
        cout << " test on k5" << std::endl;
        start = seconds();
        k5<<<grid,block>>>(c);
    
        cudaDeviceSynchronize();
        elaps = seconds() - start;
        cout << " k5 finished with ts: " << elaps << std::endl;
    
    } else {
        cout << "unknown test type should be 1 to 5" << std::endl;
    }

    cudaFree(c);
    cudaDeviceReset();

    
}