#include "common.h"

#include <cuda_runtime.h>
#include <cstdio>


using namespace std;
void initD(float* ip, const int size) {
    int i;
    for(i=0;i<size;i++)
     {
         ip[i] = (float)(rand() & 0xFF) / 10.0f;
     }
}


void sumMatrixHost(float* A, float* B, float* C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for(int iy = 0;iy<ny; iy++) {
        for(int ix = 0;ix<nx;ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
    return;
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



__global__ void sumMatrixGpu(float* A, float* B, float* C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y  + threadIdx.y;

    unsigned int idx  = iy * NX + ix;

    if(ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}



int main(int argc, char** argv) {
    int dev = 0;

    getCudaInfo();


    //NOTE. according to experiments
    //   1 nx=1<<14, ny = 1<<14 will cause error and compute result 
    //    mismatch and CudaGetLastError return 77
    //   3 nx=1<<13, ny = 1<<14 will work 
    //    execution result under this configuration:
    //       Using device : GeForce GTX 1050 Ti with Max-Q Design
    //       host sum cost: 0.321487
    //       sum matrix on gpu grid x: 256 grid y : 512 block x: 32 block y: 32 cost ts: 0.0161889
    //       res of get last error: 0

    //   scale smaller than 2 will work also
    int nx = 1<<14;
    int ny = 1<<14;

    int nxy = nx*ny;

    int nb = nxy * sizeof(float);

    float * ha, *hb, *host, *gpu;
    ha = (float*)malloc(nb);
    hb = (float*)malloc(nb);
    host = (float*) malloc(nb);
    gpu = (float*) malloc(nb);

    double start = seconds();

    initD(ha, nxy);
    initD(hb, nxy);
    
    double elaps = seconds() - start;
    memset(host, 0, nb);
    memset(gpu, 0, nb);


    start  = seconds();
    sumMatrixHost(ha, hb, host,nx, ny);

    elaps = seconds() - start;
    std::cout << " host sum cost: " << elaps << std::endl;


    float * d_matA, * d_matB, * d_matC;
    cudaMalloc((void**)&d_matA, nb) ;
    cudaMalloc((void**)&d_matB, nb) ;
    cudaMalloc((void**)&d_matC, nb) ;
    
    auto m = cudaGetLastError();
    // if m == 2 then malloc failed.
    cout << "res of get last error after cuda Malloc: " << m << std::endl;
    //start = seconds();

    cudaMemcpy(d_matA, ha, nb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, hb, nb, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;

    if(argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x -1)/block.x, (ny+block.y-1)/block.y);
    
    cudaDeviceSynchronize();
    
    start = seconds();
    sumMatrixGpu<<<grid, block>>>(d_matA, d_matB, d_matC, nx, ny);
    cudaDeviceSynchronize();

    elaps = seconds() - start;
    cout << " sum matrix on gpu grid x: " << grid.x << " grid y : " << grid.y 
         <<   " block x: " << block.x  << " block y: " << block.y << " cost ts: " 
         << elaps << std::endl;

    auto x = cudaGetLastError();
    cout << "res of get last error: " <<x << std::endl;

    cudaMemcpy(gpu, d_matC, nb, cudaMemcpyDeviceToHost);
    checkRes(host, gpu, nxy);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    free(ha);
    free(hb);
    free(host);
    free(gpu);
    cudaDeviceReset();




}