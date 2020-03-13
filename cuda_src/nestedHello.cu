#include <cstdio>
#include <cuda_runtime.h>

__global__ void NestHw(int const size, int depth) {
    printf("xxxx");
    int tid = threadIdx.x;

    printf("Rec=%d: hello world from thread %d block %d\n",depth, tid, blockIdx.x );

    if(size == 1)
        return;

    int n_threads = size>>1;
    if(tid==0 && n_threads >0)
    {
        NestHw<<<1, n_threads>>>(n_threads, ++depth);
        printf("->>>> nested exec depth: %d\n", depth);
    }
}

int main(int argc ,char** argv) {
    int size = 8;
    int b_size = 8;
    int igrid = 1;


    /*if(argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * b_size;
    }*/

    dim3 block(b_size, 1);
    dim3 grid((size + block.x - 1)/block.x, 1);

    printf("begin nest hw block.x: %d.\n", block.x);
    printf("begin nest hw grid.x: %d.\n", grid.x);
    NestHw<<<grid, block>>>(block.x, 0);
    cudaDeviceSynchronize();
    printf("end nest hw.\n");
}
