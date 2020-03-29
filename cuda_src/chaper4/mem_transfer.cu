#include "../common.h"
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    auto prop = getCudaInfo();
    
    unsigned int size = 1<20;
    unsigned int nb = size * sizeof(float);



    float* ha, *da;
    ha = (float*)malloc(nb);

    CHECK(cudaMalloc((float**)&da, nb));
    for(int i =0;i<size;i++) {
        ha[i] = 100.1f;
    }

    auto start = seconds();
    CHECK(cudaMemcpy(da, ha, nb, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy( ha, da,nb, cudaMemcpyDeviceToHost));
    auto elaps = seconds() - start;

    cout << "for array with size: " << size  << " copy from and to cost: " << elaps << endl;


    float* pha;
    CHECK(cudaMallocHost((float**)&pha, nb));

    for(int i=0;i<size;i++)  {
        pha[i] = 100.2f;
    }

    start = seconds();
    CHECK(cudaMemcpy(da, pha, nb, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(pha, da, nb, cudaMemcpyDeviceToHost));
    elaps = seconds() - start;

    cout << "for array with size: " << size  << " copy from and to cost:  (with pin host mem)" << elaps << endl;
    // it seems there is no much differences between above two parts of `memcpy` example
    // maybe this example is too simple

    CHECK(cudaFree(da));
    free(ha);
    CHECK(cudaFreeHost(pha));

    cudaDeviceReset();


}