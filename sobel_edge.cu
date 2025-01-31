#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Sobel edge detection kernel
__global__ void sobelEdge(unsigned char* input, unsigned char* output, 
                         int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip border pixels and bounds check
    if (row < 1 || row >= height-1 || col < 1 || col >= width-1) return;
    
    // Sobel operators
    int Gx = 
        -1 * input[(row-1)*width + (col-1)] +
         0 * input[(row-1)*width + col] +
         1 * input[(row-1)*width + (col+1)] +
        -2 * input[row*width + (col-1)] +
         0 * input[row*width + col] +
         2 * input[row*width + (col+1)] +
        -1 * input[(row+1)*width + (col-1)] +
         0 * input[(row+1)*width + col] +
         1 * input[(row+1)*width + (col+1)];
        
    int Gy = 
        -1 * input[(row-1)*width + (col-1)] +
        -2 * input[(row-1)*width + col] +
        -1 * input[(row-1)*width + (col+1)] +
         0 * input[row*width + (col-1)] +
         0 * input[row*width + col] +
         0 * input[row*width + (col+1)] +
         1 * input[(row+1)*width + (col-1)] +
         2 * input[(row+1)*width + col] +
         1 * input[(row+1)*width + (col+1)];
    
    // Calculate magnitude
    float magnitude = sqrtf(Gx*Gx + Gy*Gy);
    
    // Normalize to 0-255
    output[row*width + col] = (unsigned char)fminf(magnitude, 255.0f);
}

// Function to print image as ASCII art
void printASCII(unsigned char* image, int width, int height) {
    const char* ascii_chars = " .:-=+*#%@";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (int)(image[i*width + j] / 25.6); // Map 0-255 to 0-9
            printf("%c", ascii_chars[idx]);
        }
        printf("\n");
    }
}

int main() {
    // Create a simple test image (a box)
    const int width = 32;
    const int height = 32;
    unsigned char* h_input = (unsigned char*)malloc(width * height);
    unsigned char* h_output = (unsigned char*)malloc(width * height);
    
    // Initialize with a box pattern
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (i > 8 && i < 24 && j > 8 && j < 24)
                h_input[i*width + j] = 255;  // White box
            else
                h_input[i*width + j] = 0;    // Black background
        }
    }
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, width * height, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    sobelEdge<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    // Get result back
    cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Original image:\n");
    printASCII(h_input, width, height);
    printf("\nEdges detected:\n");
    printASCII(h_output, width, height);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}
