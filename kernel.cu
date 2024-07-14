#define GLEW_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "stb_image.h"
#include "stb_image_write.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>


#define WIDTH 800
#define HEIGHT 600
#define LANCZOS_A 3

__device__ float lanczosWeight(float x) {
    if (x == 0.0f) return 1.0f;
    if (x < -LANCZOS_A || x > LANCZOS_A) return 0.0f;
    x *= M_PI;
    return LANCZOS_A * sinf(x) * sinf(x / LANCZOS_A) / (x * x);
}

__global__ void lanczosUpscaleKernel(uchar4* output, const uchar4* input, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight) {
        float inputX = (float)x * inputWidth / outputWidth;
        float inputY = (float)y * inputHeight / outputHeight;

        int x1 = (int)floorf(inputX) - LANCZOS_A + 1;
        int y1 = (int)floorf(inputY) - LANCZOS_A + 1;

        float4 sum = make_float4(0, 0, 0, 0);
        float totalWeight = 0.0f;

        for (int j = 0; j < 2 * LANCZOS_A; ++j) {
            for (int i = 0; i < 2 * LANCZOS_A; ++i) {
                int ix = min(max(x1 + i, 0), inputWidth - 1);
                int iy = min(max(y1 + j, 0), inputHeight - 1);

                float weight = lanczosWeight(inputX - (x1 + i)) * lanczosWeight(inputY - (y1 + j));
                uchar4 pixel = input[iy * inputWidth + ix];

                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
                sum.z += pixel.z * weight;
                sum.w += pixel.w * weight;
                totalWeight += weight;
            }
        }

        output[y * outputWidth + x] = make_uchar4(sum.x / totalWeight, sum.y / totalWeight, sum.z / totalWeight, sum.w / totalWeight);
    }
}

std::string generateFilename(const std::string& originalFilename) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M");

    std::string timestamp = ss.str();
    std::string basename = originalFilename.substr(0, originalFilename.find_last_of("."));

    return basename + "_upscaled_" + timestamp + ".png";
}

int main()
{
    // Load image using stb_image
    int inputWidth, inputHeight, channels;
    std::string inputFilename = "C:/Users/lilyx/source/repos/cudaHackathon2/nature.jpg";
    unsigned char* imageData = stbi_load(inputFilename.c_str(), &inputWidth, &inputHeight, &channels, 4);
    if (!imageData) {
        fprintf(stderr, "Failed to load image\n");
        return -1;
    }

    int outputWidth = inputWidth * 5;  // 5x upscale in width
    int outputHeight = inputHeight;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(outputWidth, outputHeight, "CUDA Image Upscale", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, outputWidth, outputHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cudaGraphicsResource* cudaTextureResource;
    cudaGraphicsGLRegisterImage(&cudaTextureResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    uchar4* d_input = nullptr;
    uchar4* d_output = nullptr;
    cudaMalloc(&d_input, inputWidth * inputHeight * sizeof(uchar4));
    cudaMalloc(&d_output, outputWidth * outputHeight * sizeof(uchar4));

    cudaMemcpy(d_input, imageData, inputWidth * inputHeight * sizeof(uchar4), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

    lanczosUpscaleKernel << <gridSize, blockSize >> > (d_output, d_input, inputWidth, inputHeight, outputWidth, outputHeight);
    cudaDeviceSynchronize();

    // Save the upscaled image
    unsigned char* h_output = new unsigned char[outputWidth * outputHeight * 4];
    cudaMemcpy(h_output, d_output, outputWidth * outputHeight * sizeof(uchar4), cudaMemcpyDeviceToHost);

    std::string outputFilename = generateFilename(inputFilename);
    stbi_write_png(outputFilename.c_str(), outputWidth, outputHeight, 4, h_output, outputWidth * 4);
    delete[] h_output;

    while (!glfwWindowShouldClose(window)) {
        cudaArray* textureArray;
        cudaGraphicsMapResources(1, &cudaTextureResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaTextureResource, 0, 0);
        cudaMemcpy2DToArray(textureArray, 0, 0, d_output, outputWidth * sizeof(uchar4), outputWidth * sizeof(uchar4), outputHeight, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaTextureResource, 0);

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);  // Flipped vertically
        glTexCoord2f(1, 1); glVertex2f(1, -1);   // Flipped vertically
        glTexCoord2f(1, 0); glVertex2f(1, 1);    // Flipped vertically
        glTexCoord2f(0, 0); glVertex2f(-1, 1);   // Flipped vertically
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaGraphicsUnregisterResource(cudaTextureResource);
    glDeleteTextures(1, &textureID);
    glfwDestroyWindow(window);
    glfwTerminate();
    cudaDeviceReset();

    stbi_image_free(imageData);

    return 0;
}