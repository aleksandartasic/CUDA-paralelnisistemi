%%cu

#include <stdio.h>

#define N 26
#define BLOCK_SIZE 4

__global__ void izracunajB(float* A, float* B)
{
    __shared__ float sharedA[BLOCK_SIZE + 2];

    int tid = threadIdx.x;
    int indeks = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (indeks < N)
    {
        sharedA[tid] = A[indeks];
        if (tid == BLOCK_SIZE-2)
        {

            sharedA[BLOCK_SIZE] = A[indeks + 2];
        }
        if (tid == BLOCK_SIZE-1)
        {

            sharedA[BLOCK_SIZE+1] = A[indeks + 2];
        }
    }
    __syncthreads();

    if (indeks < N - 2)
    {

        B[indeks] =  (sharedA[tid] * sharedA[tid + 1] * sharedA[tid + 2]) / (sharedA[tid] + sharedA[tid + 1] + sharedA[tid + 2]);
    }
}

int main()
{
    float A[27] = {1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23, 24,25,26,27};
    float B[27 - 2];

    float* d_A;
    float* d_B;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, (N - 2) * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    int numBlocks = (N - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    izracunajB<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B);

    cudaMemcpy(B, d_B, (N - 2) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("B = ");
    for (int i = 0; i < N - 2; i++)
    {
        printf("%f ", B[i]);
    }
    printf("\n");
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
