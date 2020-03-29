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
        __syncthreads();
        //cudaDeviceSynchronize();
        
    }
    __syncthreads();

}


__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // start of this block data.
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    for(int stride  = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if( index < blockDim.x ) {
            idata[index] += idata[index+stride];
        }
        __syncthreads();
    }

    if(tid==0) g_odata[blockIdx.x]  = idata[0];


}


__global__ void reduceInterleave(int * g_idata, int * g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This is the start of data of this block.
    int *idata  = g_idata + blockIdx.x * blockDim.x;

    if(idx >=n) return;


    for(int stride = blockDim.x/2;stride > 0; stride >>= 1) {
        if(tid <stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x]  = idata[0];

}

__global__ void reduceUnrolling2(int* g_idata, int * g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    // every second block.
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    // merge block level data first.
    if (idx  + blockDim.x < n) 
        g_idata[idx ] += g_idata[idx+blockDim.x];
    
    __syncthreads();

    for(int stride  = blockDim.x /2; stride>0;stride >>=1 ) {
        if(tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid ==0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling4(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4+ tid;

    int *idata = g_idata + blockIdx * blockDim.x * 4;

    if((idx+3*blockDim.x )< n) {
        int a1 = g_idata[idx] = g_idata[idx]
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        g_idata[idx]  = a1 + a2 + a3 + a4;
    }

    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if(tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceUnrolling8(int * g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + tid;

    int *idata = g_idata + blockIdx * blockDim.x * 8;

    if((idx+7*blockDim.x )< n) {
        int a1 = g_idata[idx] = g_idata[idx]
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx]  = a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }

    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if(tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}


__global__ void reduceUnrollWarps8 (int * g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;


    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx + 7 *blockDim.x < n) {
        int a1 = g_idata[idx] = g_idata[idx]
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[idx]  = a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();


    for(int stride  = blockDim.x/2; stride >32; stride>>=1) {
        if(tid < stride) {
            idata[tid] += idata[tid+stride];
        }
    }

    if(tid < 32) {
        volatile int * vmem = idata;

        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if(tid == 0) 
    g_odata[blockIdx.x] = idata[0];
}



__global__ void gpuRecursiveReduce2(int* g_idata, int * g_odata, int stride, int const dim) {
    
    int *idata = g_idata + blockIdx.x * dim;

    if(stride == 1 && threadIdx.x == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    idata[threadIdx.x] += idata[threadIdx.x + stride];
    if(threadIdx.x ==0 && blockIdx.x == 0) {
        gpuRecursiveReduce2<<<gridDim.x, stride/2>>>(g_idata, g_odata, stride/2, dim);
    }

}



__global__ void gpuRecursiveReduceNoSync(int *g_idata, int* g_odata,
unsigned int size) {
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x* blockDim.x;
    int *odata = &g_odata[blockIdx.x];


    if(size == 2 && tid ==0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }



    int stride  = size >> 1;

    if(stride > 1 && tid < stride) {
        idata[tid] +=idata[tid+stride];
        if(tid == 0) {
            gpuRecursiveReduceNoSync<<<1, stride>>>(idata, odata,stride);
        }
    }
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

    memset(h_odata, 0, sizeof(h_odata));
    cudaMemcpy(d_odata, h_odata, sizeof(d_odata), cudaMemcpyHostToDevice);

    iStart = seconds();
    gpu_sum = 0;
    gpuRecursiveReduceNoSync<<<grid,block>>>(d_idata, d_odata, size);
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++) {
        gpu_sum += h_odata[i];
    }

    iElaps = seconds() - iStart;
    cout << " gpu recursive no sync elaps: " << iElaps << " gpu sum: " << gpu_sum 
    << " grid " << grid.x << " block " << block.x  << " Elaps: " << iElaps << endl;

    memset(h_odata, 0, sizeof(h_odata));
    cudaMemcpy(d_odata, h_odata, sizeof(d_odata), cudaMemcpyHostToDevice);

    iStart = seconds();
    gpu_sum = 0;
    gpuRecursiveReduce2<<<grid,block.x/2>>>(d_idata, d_odata, block.x/2, block.x);
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++) {
        gpu_sum += h_odata[i];
    }

    iElaps = seconds() - iStart;
    cout << " gpu recursive reduce 2  elaps: " << iElaps << " gpu sum: " << gpu_sum 
    << " grid " << grid.x << " block " << block.x  << " Elaps: " << iElaps << endl;



    memset(h_odata, 0, sizeof(h_odata));
    cudaMemcpy(d_odata, h_odata, sizeof(d_odata), cudaMemcpyHostToDevice);

    iStart = seconds();
    gpu_sum = 0;
    reduceNeighboredLess<<<grid,block>>>(d_idata, d_odata, size);
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++) {
        gpu_sum += h_odata[i];
    }

    iElaps = seconds() - iStart;
    cout << " gpu neighbor less warp divergence elaps: " << iElaps << " gpu sum: " << gpu_sum 
    << " grid " << grid.x << " block " << block.x  << " Elaps: " << iElaps << endl;

}
