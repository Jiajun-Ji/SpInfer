/***************************************************************************
 * SpInfer 自定义数据使用示例
 * 展示如何从特定数据到格式转换再到计算的完整流程
 ***************************************************************************/
#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "build/SpMM_API.cuh"
#include "kernel_benchmark/Flashllm_utils.cuh"

// 性能测试常量
#define WARM_UP_ITERATION 3
#define BENCHMARK_ITERATION 10

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 性能对比工具函数
void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error = 0.0) {
    printf("%-15s -> Time: %6.3f ms, Performance: %6.2f TFLOPS",
           KernelName, milliseconds, tflops);
    if (error > 0.0) {
        printf(", Error: %.2e", error);
    }
    printf("\n");
}

double ComputeTotalError(half* reference, half* result, int M, int N) {
    double totalError = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = __half2float(reference[i]) - __half2float(result[i]);
        totalError += diff * diff;
    }
    return sqrt(totalError);
}

// 1. 自定义数据生成函数
void generate_custom_sparse_matrix(half* A_h, int M, int K, float sparsity_ratio) {
    printf("生成自定义稀疏矩阵: M=%d, K=%d, 稀疏度=%.2f\n", M, K, sparsity_ratio);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float rand_val = static_cast<float>(rand()) / RAND_MAX;
            if (rand_val > sparsity_ratio) {
                // 非零元素：生成[-1, 1]范围的随机值
                float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
                A_h[i * K + j] = __float2half(val);
            } else {
                // 零元素
                A_h[i * K + j] = __float2half(0.0f);
            }
        }
    }
}

// 2. 从文件加载稠密矩阵
bool load_dense_matrix_from_file(const char* filename, half* A_h, int M, int K) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("无法打开文件: %s\n", filename);
        return false;
    }

    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size_t expected_size = M * K * sizeof(half);
    size_t file_elements = file_size / sizeof(half);

    if (file_size != expected_size) {
        printf("文件大小不匹配: 期望 %zu 字节 (%d x %d), 实际 %ld 字节 (%zu 元素)\n",
               expected_size, M, K, file_size, file_elements);
        fclose(file);
        return false;
    }

    size_t elements_read = fread(A_h, sizeof(half), M * K, file);
    fclose(file);

    if (elements_read != M * K) {
        printf("文件读取错误: 期望 %d 个元素，实际读取 %zu 个\n", M * K, elements_read);
        return false;
    }

    printf("成功从文件 %s 加载 %d x %d 矩阵\n", filename, M, K);
    return true;
}

// 3. 生成稠密矩阵B
void generate_dense_matrix_B(half* B_h, int K, int N) {
    for (int i = 0; i < K * N; i++) {
        float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        B_h[i] = __float2half(val);
    }
}

// 4. 主函数：完整的数据处理流程
int main(int argc, char** argv) {
    // 参数设置
    int M_GLOBAL = 1024;  // 可以根据需要修改
    int K_GLOBAL = 4096;
    int N_GLOBAL = 256;
    int Split_K = 1;
    float sparsity_ratio = 0.8f;  // 80% 稀疏度
    
    if (argc >= 4) {
        M_GLOBAL = atoi(argv[1]);
        K_GLOBAL = atoi(argv[2]);
        N_GLOBAL = atoi(argv[3]);
    }
    if (argc >= 5) {
        Split_K = atof(argv[4]);
        sparsity_ratio = atof(argv[5]);
    }
    
    printf("=== SpInfer 自定义数据处理示例 ===\n");
    printf("矩阵维度: A(%d x %d) × B(%d x %d) = C(%d x %d)\n", 
           M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
    
    // ========== 步骤1: 准备输入数据 ==========
    printf("\n步骤1: 准备输入数据\n");
    
    // 分配host内存
    half* A_h = (half*)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    half* B_h = (half*)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    
    if (!A_h || !B_h) {
        printf("Host内存分配失败\n");
        return -1;
    }
    
    // 方式1: 从Python生成器加载自定义稀疏矩阵
    if (!load_dense_matrix_from_file("Input_date/sparse_data_latest/custom_sparse_matrix.bin", A_h, M_GLOBAL, K_GLOBAL)) {
        printf("未找到自定义稀疏矩阵文件，使用随机生成\n");
        generate_custom_sparse_matrix(A_h, M_GLOBAL, K_GLOBAL, sparsity_ratio);
    } else {
        printf("成功加载Python生成的自定义稀疏矩阵\n");
    }

    // 方式2: 从其他文件加载（可选）
    // if (!load_dense_matrix_from_file("your_matrix.bin", A_h, M_GLOBAL, K_GLOBAL)) {
    //     printf("使用随机生成的矩阵\n");
    //     generate_custom_sparse_matrix(A_h, M_GLOBAL, K_GLOBAL, sparsity_ratio);
    // }
    
    // 生成稠密矩阵B
    generate_dense_matrix_B(B_h, K_GLOBAL, N_GLOBAL);
    
    // ========== 步骤2: 稀疏格式转换 ==========
    printf("\n步骤2: 稀疏格式转换\n");
    
    // 调用SpInfer的bitmap格式转换
    half* Compressed_Val_cpu = nullptr;
    int* TileOffsets_cpu = nullptr;
    int* TileOffsets_median_cpu = nullptr;
    int* TileOffsets_global_cpu = nullptr;
    uint64_t* bitmap_cpu = nullptr;
    int max_nnz_intile = 0;
    
    int num_global_tiles = InitSparseMatrixA_bitmap(
        A_h, M_GLOBAL, K_GLOBAL,
        8, 16, 64,    // tile_M, tile_M_median, tile_M_global
        8, 64, 64,    // tile_K, tile_K_median, tile_K_global
        &Compressed_Val_cpu,
        &TileOffsets_cpu,
        &TileOffsets_median_cpu,
        &TileOffsets_global_cpu,
        &bitmap_cpu,
        max_nnz_intile
    );
    
    // 计算各层tile数量
    int local_tile_num = 8 * 8;      // 64个小块per大块
    int median_tile_num = 4 * 1;     // 4个中块per大块
    int num_local_tiles = num_global_tiles * local_tile_num;
    int num_median_tiles = num_global_tiles * median_tile_num;
    
    int val_count = TileOffsets_global_cpu[num_global_tiles];
    
    // 调整max_nnz_intile为64的倍数
    if (max_nnz_intile % 64 != 0) {
        max_nnz_intile = ((max_nnz_intile / 64) + 1) * 64;
    }
    
    printf("转换完成: %d个全局块, %d个非零元素, 最大块内非零元素: %d\n", 
           num_global_tiles, val_count, max_nnz_intile);
    
    // ========== 步骤3: 数据传输到GPU ==========
    printf("\n步骤3: 数据传输到GPU\n");
    
    // 分配GPU内存
    half* A_gpu, *B_gpu, *C_gpu;
    half* Compressed_Val_gpu;
    int* TileOffsets_global_gpu, *TileOffsets_median_gpu;
    uint64_t* bitmap_gpu;
    int* max_nnz_intile_gpu;
    half* Reduction_Workspace;
    
    CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(half) * M_GLOBAL * K_GLOBAL));
    CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(half) * K_GLOBAL * N_GLOBAL));
    CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(half) * M_GLOBAL * N_GLOBAL));
    CHECK_CUDA(cudaMalloc(&Compressed_Val_gpu, sizeof(half) * std::max(val_count, 1)));
    CHECK_CUDA(cudaMalloc(&TileOffsets_global_gpu, sizeof(int) * (num_global_tiles + 1)));
    CHECK_CUDA(cudaMalloc(&TileOffsets_median_gpu, sizeof(int) * num_median_tiles));
    CHECK_CUDA(cudaMalloc(&bitmap_gpu, sizeof(uint64_t) * num_local_tiles));
    CHECK_CUDA(cudaMalloc(&max_nnz_intile_gpu, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&Reduction_Workspace, sizeof(half) * M_GLOBAL * N_GLOBAL * Split_K));
    
    // 拷贝数据到GPU
    CHECK_CUDA(cudaMemcpy(A_gpu, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_gpu, B_h, sizeof(half) * K_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(Compressed_Val_gpu, Compressed_Val_cpu, sizeof(half) * val_count, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(TileOffsets_global_gpu, TileOffsets_global_cpu, sizeof(int) * (num_global_tiles + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(TileOffsets_median_gpu, TileOffsets_median_cpu, sizeof(int) * num_median_tiles, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bitmap_gpu, bitmap_cpu, sizeof(uint64_t) * num_local_tiles, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(max_nnz_intile_gpu, &max_nnz_intile, sizeof(int), cudaMemcpyHostToDevice));
    
    printf("数据传输完成\n");
    
    // ========== 步骤4: 多库性能对比测试 ==========
    printf("\n步骤4: 多库性能对比测试\n");

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 计算性能指标的基础数据
    long long ops = 2LL * M_GLOBAL * N_GLOBAL * K_GLOBAL;  // 乘加操作数

    printf("\n=== 性能对比结果 ===\n");

    // 1. cuBLAS (Tensor Core) 性能测试
    printf("测试 cuBLAS (Tensor Core)...\n");
    half* C_cublas_gpu = NULL;
    CHECK_CUDA(cudaMalloc(&C_cublas_gpu, sizeof(half) * M_GLOBAL * N_GLOBAL));
    CHECK_CUDA(cudaMemset(C_cublas_gpu, 0, sizeof(half) * M_GLOBAL * N_GLOBAL));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // 启用Tensor Core

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS预热
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     M_GLOBAL, N_GLOBAL, K_GLOBAL,
                     &alpha, A_gpu, CUDA_R_16F, K_GLOBAL,
                     B_gpu, CUDA_R_16F, K_GLOBAL,
                     &beta, C_cublas_gpu, CUDA_R_16F, M_GLOBAL,
                     CUDA_R_32F, static_cast<cublasGemmAlgo_t>(0));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // cuBLAS计时
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     M_GLOBAL, N_GLOBAL, K_GLOBAL,
                     &alpha, A_gpu, CUDA_R_16F, K_GLOBAL,
                     B_gpu, CUDA_R_16F, K_GLOBAL,
                     &beta, C_cublas_gpu, CUDA_R_16F, M_GLOBAL,
                     CUDA_R_32F, static_cast<cublasGemmAlgo_t>(0));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds_cublas = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds_cublas, start, stop));
    milliseconds_cublas /= BENCHMARK_ITERATION;
    float tflops_cublas = (ops / (milliseconds_cublas / 1000.0)) / 1e12;

    // 2. SpInfer 性能测试
    printf("测试 SpInfer...\n");
    CHECK_CUDA(cudaMemset(C_gpu, 0, sizeof(half) * M_GLOBAL * N_GLOBAL));

    // SpInfer预热
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        SpMM_SplitK_API_bitmap_v3(0, A_gpu, Compressed_Val_gpu,
                                  TileOffsets_global_gpu, TileOffsets_median_gpu,
                                  bitmap_gpu, max_nnz_intile_gpu, B_gpu, C_gpu,
                                  M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // SpInfer计时
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATION; i++) {
        SpMM_SplitK_API_bitmap_v3(0, A_gpu, Compressed_Val_gpu,
                                  TileOffsets_global_gpu, TileOffsets_median_gpu,
                                  bitmap_gpu, max_nnz_intile_gpu, B_gpu, C_gpu,
                                  M_GLOBAL, N_GLOBAL, K_GLOBAL, Reduction_Workspace, Split_K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds_spinfer = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds_spinfer, start, stop));
    milliseconds_spinfer /= BENCHMARK_ITERATION;
    float tflops_spinfer = (ops / (milliseconds_spinfer / 1000.0)) / 1e12;
    
    // ========== 步骤5: 结果验证和性能对比 ==========
    printf("\n步骤5: 结果验证和性能对比\n");

    // 将结果拷贝回host进行验证
    half* C_spinfer_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
    half* C_cublas_h = (half*)malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);

    CHECK_CUDA(cudaMemcpy(C_spinfer_h, C_gpu, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C_cublas_h, C_cublas_gpu, sizeof(half) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    // 计算误差
    double error_spinfer = ComputeTotalError(C_cublas_h, C_spinfer_h, M_GLOBAL, N_GLOBAL);

    // 输出性能对比结果
    printf("\n=== 性能对比总结 ===\n");
    PrintPerformance("cuBLAS (TC)", milliseconds_cublas, tflops_cublas);
    PrintPerformance("SpInfer", milliseconds_spinfer, tflops_spinfer, error_spinfer);

    // 计算加速比
    float speedup = milliseconds_cublas / milliseconds_spinfer;
    printf("\nSpInfer vs cuBLAS 加速比: %.2fx\n", speedup);

    // 简单验证：检查结果是否包含非零值
    int non_zero_count = 0;
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        if (__half2float(C_spinfer_h[i]) != 0.0f) {
            non_zero_count++;
        }
    }
    printf("SpInfer结果矩阵中非零元素数量: %d / %d\n", non_zero_count, M_GLOBAL * N_GLOBAL);

    // 清理cuBLAS资源
    cublasDestroy(handle);
    cudaFree(C_cublas_gpu);
    free(C_cublas_h);
    
    // ========== 清理资源 ==========
    printf("\n清理资源\n");
    
    // 释放host内存
    free(A_h);
    free(B_h);
    free(C_spinfer_h);
    free(Compressed_Val_cpu);
    free(TileOffsets_cpu);
    free(TileOffsets_median_cpu);
    free(TileOffsets_global_cpu);
    free(bitmap_cpu);
    
    // 释放GPU内存
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(Compressed_Val_gpu);
    cudaFree(TileOffsets_global_gpu);
    cudaFree(TileOffsets_median_gpu);
    cudaFree(bitmap_gpu);
    cudaFree(max_nnz_intile_gpu);
    cudaFree(Reduction_Workspace);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("程序执行完成!\n");
    return 0;
}
