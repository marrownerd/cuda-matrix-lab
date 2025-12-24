#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

// Макросы для проверки ошибок
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

//Global Memory)
__global__ void matrixMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Алгоритм с разделяемой памятью (Shared Memory)
__global__ void matrixMulShared(const float *A, const float *B, float *C, int N) {
    // Выделяем память на блок
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Проходим по всем плиткам (тайлам)
    // m - индекс плитки
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // Загрузка данных в Shared Memory
        // Проверка границ обязательна
        if (row < N && m * BLOCK_SIZE + tx < N)
            sA[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;

        if (col < N && m * BLOCK_SIZE + ty < N)
            sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        // Ждем, пока все потоки блока загрузят данные
        __syncthreads();

        // Вычисляем частичную сумму для этой плитки
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        // Ждем, пока все закончат считать, перед загрузкой новой плитки
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Вспомогательные функции (CPU)

void cpuMatrixMul(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void initMatrix(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = ((float)rand() / RAND_MAX);
    }
}

bool checkResult(const float *ref, const float *gpu, int size) {
    double epsilon = 1.0e-3;
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - gpu[i]) > epsilon) {
            printf("Error at index %d: CPU=%f, GPU=%f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main() {
    // Размеры матриц для тестирования
    int sizes[] = {128, 256, 512, 1024}; 
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);

    printf("========================================================================\n");
    printf("СРАВНЕНИЕ МЕТОДОВ УМНОЖЕНИЯ МАТРИЦ (ВРЕМЯ В МС)\n");
    printf("GPU: NVIDIA CUDA\n");
    printf("Block Size: %dx%d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("========================================================================\n");
    printf("%-10s | %-10s | %-10s | %-10s | %-10s | %s\n", 
           "Размер", "CPU", "Naive", "Shared", "cuBLAS", "Status");
    printf("------------------------------------------------------------------------\n");

    for (int t = 0; t < num_tests; ++t) {
        int N = sizes[t];
        size_t bytes = N * N * sizeof(float);

        // Хост память
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_CPU = (float*)malloc(bytes);
        float *h_C_GPU = (float*)malloc(bytes);

        initMatrix(h_A, N * N);
        initMatrix(h_B, N * N);

        // Девайс память
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        // События для тайминга
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float ms_naive = 0, ms_shared = 0, ms_cublas = 0, ms_cpu = 0;

        // 1. CPU (Только для малых матриц, иначе очень долго)
        if (N <= 1024) { // Ограничим CPU тест, иначе на 2048+ будем ждать вечность
            clock_t cpu_start = clock();
            cpuMatrixMul(h_A, h_B, h_C_CPU, N);
            clock_t cpu_end = clock();
            ms_cpu = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
        } else {
            ms_cpu = -1.0f; // Skip
        }

        // Настройка сетки
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // 2. Naive CUDA
        CUDA_CHECK(cudaEventRecord(start));
        matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
        
        // Проверка Naive
        CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost));
        bool naive_ok = (N <= 1024) ? checkResult(h_C_CPU, h_C_GPU, N * N) : true;

        // 3. Shared Memory CUDA
        CUDA_CHECK(cudaMemset(d_C, 0, bytes)); // Очистка результата
        CUDA_CHECK(cudaEventRecord(start));
        matrixMulShared<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

        // Проверка Shared
        CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost));
        bool shared_ok = (N <= 1024) ? checkResult(h_C_CPU, h_C_GPU, N * N) : true;

        // 4. cuBLAS
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f;
        float beta = 0.0f;

        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        
        // Трюк с порядком аргументов (B, A) для Row-Major матриц
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, start, stop));
        CUBLAS_CHECK(cublasDestroy(handle));

        // Проверка cuBLAS
        CUDA_CHECK(cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost));
        bool cublas_ok = (N <= 1024) ? checkResult(h_C_CPU, h_C_GPU, N * N) : true;

        // Вывод строки таблицы
        char status[20] = "FAIL";
        if (naive_ok && shared_ok && cublas_ok) sprintf(status, "OK");

        printf("%dx%d    | %-10.3f | %-10.3f | %-10.3f | %-10.3f | %s\n", 
               N, N, ms_cpu, ms_naive, ms_shared, ms_cublas, status);

        // Очистка памяти
        free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
    return 0;
}