#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define CHANNELS 3
#define BLOCK_SIZE 32

const char* inputImage = "../res/input.png";
const char* outputImage = "../res/output.png";

unsigned int* loadImageData (const unsigned char* img, unsigned int size);

unsigned char* saveImageData (const unsigned int* img, unsigned int size);

__global__ void erosionKernel(const unsigned int* input, unsigned int* output, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // center pixel
        int pixel = (int)input[x + y * width];

        // left pixel
        if (x > 0 && input[(x - 1) + y * width] < pixel)
            pixel = (int)input[(x - 1) + y * width];

        // right pixel
        if (x < width - 1 && input[(x + 1) + y * width] < pixel)
            pixel = (int)input[(x + 1) + y * width];

        // top pixel
        if (y > 0 && input[x + (y - 1) * width] < pixel)
            pixel = (int)input[x + (y - 1) * width];

        // bottom pixel
        if (y < height - 1 && input[x + (y + 1) * width] < pixel)
            pixel = (int)input[x + (y + 1) * width];

        // save data
        output[x + y * width] = pixel;
    }
}

int main() {
    int width;
    int height;
    int rgb;
    int depth = 1;

    std::cout << "Enter erosion depth: ";
    std::cin >> depth;

    unsigned char* image = stbi_load(inputImage, &width, &height, &rgb, CHANNELS);
    unsigned int* hostInput = loadImageData(image, width * height * CHANNELS);
    unsigned int* hostOutput = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    unsigned int* deviceInput;
    unsigned int* deviceOutput;

    unsigned int imageSize = width * height * sizeof(unsigned int);

    cudaMalloc((void**)&deviceInput, imageSize);
    cudaMalloc((void**)&deviceOutput, imageSize);
    cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    for (unsigned int i = 0; i < depth; ++i) {
        erosionKernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height);
        cudaMemcpy(deviceInput, deviceOutput, imageSize, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_png(outputImage, width, height, CHANNELS,
                   saveImageData(hostOutput, width * height), width * CHANNELS);

    stbi_image_free(image);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    free(hostInput);
    free(hostOutput);

    return 0;
}

unsigned int* loadImageData (const unsigned char* img, unsigned int size) {
    unsigned int* data = new unsigned int[size / CHANNELS];

    int sum = 0;

    for (unsigned int i = 0; i < size; ++i) {
        sum += img[i];
        if ((i + 1) % CHANNELS == 0) {
            data[i / CHANNELS] = sum > 0 ? 1 : 0;
            sum = 0;
        }
    }

    return data;
}

unsigned char* saveImageData (const unsigned int* img, unsigned int size) {
    unsigned char* data = new unsigned char[size * CHANNELS];

    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < CHANNELS; ++j) {
            data[i * CHANNELS + j] = img[i] * 255;
        }
    }

    return data;
}
