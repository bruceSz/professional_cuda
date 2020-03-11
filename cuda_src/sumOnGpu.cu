#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>

class TsCounter{
    private: 
    time_t start;
    public:

    TsCounter() {
        printf("ts counter start.\n");
        start = time(NULL);
    }
    ~TsCounter() {
        time_t curr = time(NULL);
        printf("ts counter end.\n");
        printf("cost ts: %f\n", difftime(curr,start));

    }

};

__global__ void sum(float * A, float *B , float *C, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
        C[idx] = A[idx] + B[idx];
    printf("caculate on block %d, thread %d with %f + %f = %f\n",blockIdx.x, 
           threadIdx.x,A[idx],B[idx],C[idx]);
}

void initData(float * ip, int size) {
    time_t t;
    srand((unsigned int)time(&t));
    for(int i=0;i<size; i++) {
        ip[i] = (float)(rand() & 0xaff) / 10.0f;
    }
}


void printA(float* arr, const int N)
{
    for(int i=0;i<N;i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}


int main() {

    int n = 1000;
    size_t n_b = n * sizeof(float);
    float *a, *b;
    float *c;

    a = (float*) malloc(n_b);
    b = (float*) malloc(n_b);
    c = (float*) malloc(n_b);

    initData(a,n);
    initData(b,n);

    TsCounter a_time;
    for(int i=0;i<n;i++) {
        c[i] = a[i] + b[i];
        printf("caculate on %d with %f + %f = %f\n", i, a[i],b[i],c[i]);
    }

}

int main1() {
    int n = 1000;
    size_t n_b = n * sizeof(float);
    float *a, *b;

    a = (float*) malloc(n_b);
    b = (float*) malloc(n_b);

    initData(a,n);
    initData(b,n);

    printf("vec a: ");
    printA(a, n);

    printf("vec b: ");
    printA(b, n);

    float* ca, * cb, *cc;

    cudaMalloc((float**)&ca, n_b);
    cudaMalloc((float**)&cb, n_b);
    cudaMalloc((float**)&cc, n_b);


    cudaMemcpy(ca, a, n_b, cudaMemcpyHostToDevice);
    cudaMemcpy(cb, b, n_b, cudaMemcpyHostToDevice);


    dim3 block(2);
    dim3 thread(n/2);
    TsCounter a_time;

    sum<<<block,thread>>>(ca, cb, cc, n);    

    free(a);
    free(b);

    cudaFree(ca);
    cudaFree(cb);
    cudaFree(cc);
    printf("end of main.");
    return 0;
    
}
