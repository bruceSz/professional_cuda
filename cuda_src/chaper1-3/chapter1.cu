

#include <stdio.h>


__global__ void HelloFromGpu() {
        int x = threadIdx.x;
        int bx = blockIdx.x;
        //if (x == 5)
            printf("hello world from gpu b:%d, thread %d \n",bx,x);
}

int main() {
	HelloFromGpu<<<10,1>>>();
	//cudaDeviceReset();
        cudaDeviceSynchronize();
	return 0;
}
