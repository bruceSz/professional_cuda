#include "../common.h"
#include <cuda_runtime.h>
#include <cstdio>

#define LEN 1<< 22

struct innerStruct
{
    float x;
    float y;
};

struct innerArray {
    float x[LEN];
    float y[LEN];
};

void initialInnerStruct(innerStruct* ip, int size) {
    for(int i=0;i<size;i++) {
        ip[i].x = (float)(rand() & 0XFF)/100.0f;
        ip[i].y = (float)(rand() & 0XFF)/100.0f;
    }

    return ;
}


void initialInnerArray(innerArray* ip, int size) {
    for(int i=0;i<size;i++) {
        ip->x[i] = (float)(rand() & 0xFF);
        ip->y[i] = (float)(rand() & 0xFF);
    }
}


void testInnerStructHost(innerStruct* A, innerStruct* C, const int n) {
    for(int idx = 0;idx<n;idx++) {
        C[idx].x = A[idx].x + 10.0f;
        C[idx].y = A[idx].y + 20.0f;
    }
    return;
}

void checkInnerStruct(innerStruct* host, innerStruct* gpu, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for(int i=0;i<n;i++) {
        if(abs(host[i].x - gpu[i].x)> epsilon) {
            match = 0;
            printf("difference on %dth element: host %f\n", i, host[i].x, gpu[i].x);
            break;
        }


        if(abs(host[i].y - gpu[i].y) > epsilon) {
            match = 0;
            printf("difference on %dth ele: host %f, gpu %f\n", i, host[i].y, gpu[i].y );
            break;
        }
    }
    if(!match) printf("Arrays do not match.\n\n");
}



__global__ void testInnerStructGpu(innerStruct* data, innerStruct* res, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.0f;
        tmp.y += 20.0f;
        res[i] = tmp;

    }
}


__global__ void warmup(innerStruct* data, innerStruct* res, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<n)
    {
        innerStruct tmp = data[i];
        tmp.x += 10.0f;
        tmp.y += 20.0f;
        res[i] = tmp;

    }
}


void testInnerArrayHost(innerArray* A, innerArray* B, const int n) {
    for(int idx = 0;idx<n;idx++) {
        B->x[idx] = A->x[idx] + 10.f;
        B->y[idx] = A->y[idx] + 20.f;

    }
    return;
}


__global__ void testInnerArrayGpu(innerArray* data, innerArray* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;

        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}





int main(int argc, char** argv) {
    auto prop = getCudaInfo();

    int size = LEN;
    size_t nb = size * sizeof(innerStruct);
    innerStruct * ha = (innerStruct*) malloc(nb);
    innerStruct * host = (innerStruct*)malloc(nb);
    innerStruct * gpu = (innerStruct*)malloc(nb);

    initialInnerStruct(ha, size);
    testInnerStructHost(ha, host, size);

    innerStruct * da, *dc;

    CHECK(cudaMalloc((innerStruct**)&da, nb));
    CHECK(cudaMalloc((innerStruct**)&dc, nb));

    CHECK(cudaMemcpy(da, ha, nb, cudaMemcpyHostToDevice));
    
    int blocksize = 128;

    if (argc > 1) blocksize  = atoi(argv[1]);
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x -1)/block.x, 1);

    auto start = seconds();
    warmup<<<grid, block>>>(da, dc, size);

    CHECK(cudaDeviceSynchronize());

    auto elaps = seconds() - start;

    cout << " warmup elaps: " << elaps << std::endl;


    start = seconds();
    testInnerStructGpu<<<grid, block>>>(da, dc, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

     elaps = seconds() - start;

    cout << " gpu elaps: " << elaps << std::endl;

    
    start = seconds();
    testInnerStructHost(ha, host, size);
    //CHECK(cudaDeviceSynchronize());

     elaps = seconds() - start;

    cout << " host elaps: " << elaps << std::endl;


    size_t nb_array = sizeof(innerArray);
    innerArray * haa = (innerArray*)malloc(nb_array);
    innerArray *hosta = (innerArray*)malloc(nb_array);
    innerArray *gpua = (innerArray*)malloc(nb_array);

    initialInnerArray(haa, size);

    innerArray * daa, * dca;
    CHECK(cudaMalloc((innerArray**)&daa, nb));
    CHECK(cudaMalloc((innerArray**)&dca , nb));

    CHECK(cudaMemcpy(daa, haa,  nb, cudaMemcpyHostToDevice));
    
    start = seconds();
    testInnerArrayHost(haa, hosta, size);

    elaps = seconds() - start;

    cout << " array host elaps: " << elaps << std::endl;


    start = seconds();

    testInnerArrayGpu<<<grid, block>>>(daa, dca, size);

    elaps = seconds() - start;
    cout << "array gpu elaps: " << elaps << endl;





}

