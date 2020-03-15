#include <cstdio>
#include <stdlib.h>
#include <iostream>
#include "./common.h"
#include <cuda_runtime.h>

using namespace std;

int cpuRecursiveReduce(int* data, int const size) {
    if (size == 1) return data[0];
    int const stride = size/2;
    for(int i=0;i<stride;i++) {
        data[i] += data[i+stride];
    }
    return cpuRecursiveReduce(data, stride);
}


__global__ void reduceNeighbored(int *g_data, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x  + threadIdx.x;

    int* i_data = g_data + blockIdx.x * blockDim.x;
    if(idx >=n) return;

    for( int stride = 1; stride < blockDim.x;stride*=2 ) {
        if((tid %( 2*stride))==0) {
            //printf("tid : %d add with stride : %d, of block: %d \n" ,tid, stride, blockIdx.x );
            i_data[tid] += i_data[tid+stride];
        }

        __syncthreads();
        
    }
    if(tid ==0) g_odata[blockIdx.x] = i_data[0];
}

__global__ void gpuRecursiveReduce(int * g_data, int * g_odata, unsigned int size) {
    unsigned int tid = threadIdx.x;
    int *idata = g_data + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if(size==2 && tid == 0) {
        g_odata[blockIdx.x]= idata[0] + idata[1];
        return ;
    }

    int stride = size >>1;
    if(stride > 1 && tid < stride) {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads(); 

    if(tid == 0) {
        gpuRecursiveReduce<<<1,stride>>>(idata, odata, stride);
        //__syncthreads();
        cudaDeviceSynchronize();
        
    }
    __syncthreads();

}


int main() {
    int dev = 0;
    int gpu_sum;

    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cout << "device " << dev << " name : " << deviceProp.name << endl;
    //cudaSetDevice(dev);

    int nblock = 2048;
    int nthread = 512;

    int size = nblock * nthread;
    
    dim3 block(nthread, 1);
    dim3 grid((size + block.x - 1)/block.x, 1);
    cout << " array " << size  << " grid " << grid.x << " block " << block.x << endl;

    cout << " 2 mod 4 " << 2%4 << endl;
    cout << " 2 mod 8:" << 0%8 << endl;

    size_t b = size * sizeof(int);

    int * h_idata = (int*) malloc(b);
    int * h_odata = (int *) malloc(grid.x * sizeof(int));
    int * tmp = (int *)malloc(b);

    
    for(int i=0;i<size;i++) {
        h_idata[i] = (int)(rand() &0xFF);
        h_idata[i] = 1;
    }

    memcpy(tmp, h_idata, b);

    int * d_idata = NULL;
    int * d_odata = NULL;

    cudaMalloc((void**)&d_idata , b);
    cudaMalloc((void**)&d_odata, grid.x * sizeof(int));

    double iStart, iElaps;

    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    cout << " cpu  sum are : " << cpu_sum << " cost ts: " << iElaps << endl;


    cudaMemcpy(d_idata, h_idata, b, cudaMemcpyHostToDevice);
    iStart = seconds();

    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);

    //cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int) , cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    
    for(int i=0;i<grid.x;i++) {
        gpu_sum += h_odata[i];
    }
    iElaps = seconds() - iStart;
    cout << "" << endl;
    cout << " gpu neighbored elaps: " << iElaps << " gpu sum: " << gpu_sum 
    << " grid " << grid.x << " block " << block.x  << " Elaps: " << iElaps << endl;

    memset(h_odata, 0, sizeof(h_odata));
    cudaMemcpy(d_odata, h_odata, sizeof(d_odata), cudaMemcpyHostToDevice);

    iStart = seconds();
    gpu_sum = 0;
    gpuRecursiveReduce<<<grid,block>>>(d_idata, d_odata, size);
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++) {
        gpu_sum += h_odata[i];
    }
    iElaps = seconds() - iStart;
    cout << " gpu recursive elaps: " << iElaps << " gpu sum: " << gpu_sum 
    << " grid " << grid.x << " block " << block.x  << " Elaps: " << iElaps << endl;



}
