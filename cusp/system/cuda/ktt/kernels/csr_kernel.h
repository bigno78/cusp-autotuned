__global__ void test_kernel(int a, int* out) {
    // int id = blockIdx.x*blockDim.x + threadIdx.x;
    // if (id == 0) {
    //     //printf("Calling kernel with %d.\n", a);
    //     out[0] = 7;
    // }
}

__global__ void vectorAddition(const float* a, const float* b, float* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    result[index] = a[index] + b[index];
}
