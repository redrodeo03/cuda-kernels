#include <iostream>

__global__ void addVec(const float* A, const float* B, float* C, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    for(int i=0; i<N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2*i);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (N+blocksize-1)/blocksize;

    addVec<<<gridsize, blocksize>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Resultant array C:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
