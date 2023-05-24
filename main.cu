#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#define BLOCK_SIZE 32
#define CHANNELS 3

#define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char* file, int line) {
    if (error_code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n",
                error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

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

    CUDACHECK(cudaMalloc((void**)&deviceInput, imageSize));
    CUDACHECK(cudaMalloc((void**)&deviceOutput, imageSize));
    CUDACHECK(cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice));

    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    for (unsigned int i = 0; i < depth; ++i) {
        erosionKernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, width, height);
        CUDACHECK(cudaPeekAtLastError()); //check if kernel failed
        CUDACHECK(cudaMemcpy(deviceInput, deviceOutput, imageSize, cudaMemcpyDeviceToDevice));
    }

    CUDACHECK(cudaMemcpy(hostOutput, deviceOutput, imageSize, cudaMemcpyDeviceToHost));

    stbi_write_png(outputImage, width, height, CHANNELS,
                   saveImageData(hostOutput, width * height), width * CHANNELS);

    stbi_image_free(image);

    CUDACHECK(cudaFree(deviceInput));
    CUDACHECK(cudaFree(deviceOutput));

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
