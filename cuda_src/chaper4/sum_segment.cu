#include "../common.h"
#include <cuda_runtime.h>
#include <cstdio>


// for the reason of this offset, warmup should be slower?
__global__ void warmup(float* A, float*B, float* C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i+offset;
    if(k<n) C[i] = A[k] + B[k];
}

__global__ void readoffsetUnroll(float* A, float* B, float* C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x*2 + threadIdx.x;
    unsigned int k = i + offset;
    if(k+blockDim.x < n) {
        C[i] = A[k] + B[k];
        C[i+blockDim.x] = A[k+blockDim.x] + B[k+blockDim.x];
    }
}


__global__ void readoffsetUnroll4(float* A, float* B, float* C, const int n, int offset) {
    unsigned int i = blockIdx.x * blockDim.x*4 + threadIdx.x;
    unsigned int k = i + offset;
    if(k+blockDim.x < n) {
        C[i] = A[k] + B[k];
        C[i+blockDim.x] = A[k+blockDim.x] + B[k+blockDim.x];
        C[i+2*blockDim.x] = A[k+2*blockDim.x] + B[k+2*blockDim.x];
        C[i+3*blockDim.x] = A[k+2*blockDim.x] + B[k+3*blockDim.x];
    }
}

int main(int argc, char** argv) {
    auto prop = getCudaInfo();
    

    int size = 1<<20;
    int nb = size * sizeof(float);
    cout << "with arrya of size: " << size << endl;


    int blocksize = 512;
    int offset = 0;


    dim3 block(blocksize, 1);
    dim3 grid((size + block.x-1)/block.x, 1);

    float *ha = (float*)malloc(nb);
    float *hb = (float*)malloc(nb);

    float *host = (float*)malloc(nb);
    float *gpu = (float*)malloc(nb);

    initD(ha, size);
    memcpy(hb, ha, nb);
    sumArrayOnHost(ha, hb, host, size/2,  offset);

    float* da, *db, *dc;
    CHECK(cudaMalloc((float**)&da, nb));
    CHECK(cudaMalloc((float**)&db, nb));
    CHECK(cudaMalloc((float**)&dc, nb));



    CHECK(cudaMemcpy(da, ha, nb, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, ha, nb, cudaMemcpyHostToDevice));

    auto start = seconds();
    // only sum first half.
    warmup<<<grid, block>>>(da, db, dc, size/2, offset);

    CHECK(cudaDeviceSynchronize());
    auto elaps = seconds() - start;

    cout << " warmup cause : " << elaps << endl;


    start = seconds();
    offset = size/2;
    // sum second half.
    warmup<<<grid, block>>>(da, db, dc, size, offset);

    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;

    cout << " sum array with offset 1000 : " << elaps << endl;


    start = seconds();
    offset = size/2;
    // sum second half.
    readoffsetUnroll<<<grid, block>>>(da, db, dc, size, offset);

    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;

    cout << " sum array with unroll: " << elaps << endl;


    start = seconds();
    offset = size/2;
    // sum second half.
    readoffsetUnroll4<<<grid, block>>>(da, db, dc, size, offset);

    CHECK(cudaDeviceSynchronize());
    elaps = seconds() - start;

    cout << " sum array with unroll 4 : " << elaps << endl;


}


