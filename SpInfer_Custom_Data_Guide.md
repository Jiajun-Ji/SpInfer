# SpInfer 自定义数据使用指南

本指南详细说明如何在SpInfer中使用自定义数据，从数据准备到格式转换再到执行计算的完整流程。

## 目录
1. [数据生成方式](#数据生成方式)
2. [格式转换流程](#格式转换流程)
3. [计算执行](#计算执行)
4. [完整示例](#完整示例)
5. [性能优化建议](#性能优化建议)

## 数据生成方式

### 方式1: 程序内生成随机稀疏矩阵

```cpp
void generate_custom_sparse_matrix(half* A_h, int M, int K, float sparsity_ratio) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float rand_val = static_cast<float>(rand()) / RAND_MAX;
            if (rand_val > sparsity_ratio) {
                // 非零元素
                float val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
                A_h[i * K + j] = __float2half(val);
            } else {
                // 零元素
                A_h[i * K + j] = __float2half(0.0f);
            }
        }
    }
}
```

### 方式2: 从二进制文件加载

```cpp
bool load_dense_matrix_from_file(const char* filename, half* A_h, int M, int K) {
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    
    size_t elements_read = fread(A_h, sizeof(half), M * K, file);
    fclose(file);
    
    return (elements_read == M * K);
}
```

### 方式3: 从Python/NumPy数组转换

```python
import numpy as np
import ctypes

# 创建稀疏矩阵
def create_sparse_matrix(M, K, sparsity_ratio):
    # 生成随机矩阵
    matrix = np.random.randn(M, K).astype(np.float16)
    
    # 应用稀疏性
    mask = np.random.random((M, K)) > sparsity_ratio
    matrix = matrix * mask
    
    return matrix

# 保存为二进制文件
def save_matrix_to_binary(matrix, filename):
    matrix.astype(np.float16).tofile(filename)

# 示例使用
M, K = 1024, 4096
sparsity = 0.8
matrix = create_sparse_matrix(M, K, sparsity)
save_matrix_to_binary(matrix, "custom_matrix.bin")
```

## 格式转换流程

### 核心转换函数

SpInfer使用`InitSparseMatrixA_bitmap`函数将稠密矩阵转换为分层稀疏格式：

```cpp
int num_global_tiles = InitSparseMatrixA_bitmap(
    A_h,                        // 输入：稠密矩阵 (M×K, row-major)
    M_GLOBAL, K_GLOBAL,         // 矩阵维度
    8, 16, 64,                  // tile_M, tile_M_median, tile_M_global
    8, 64, 64,                  // tile_K, tile_K_median, tile_K_global
    &Compressed_Val_cpu,        // 输出：压缩的非零值数组
    &TileOffsets_cpu,           // 输出：小块(8×8)偏移数组
    &TileOffsets_median_cpu,    // 输出：中块(16×64)偏移数组
    &TileOffsets_global_cpu,    // 输出：大块(64×64)偏移数组
    &bitmap_cpu,                // 输出：bitmap数组(每个8×8块一个64位bitmap)
    max_nnz_intile              // 输出：单个大块内最大非零元素数
);
```

### 输出数据结构说明

1. **Compressed_Val_cpu**: 所有非零元素的压缩存储
2. **bitmap_cpu**: 每个8×8小块的64位bitmap，标记非零元素位置
3. **TileOffsets_cpu**: 每个小块的非零元素数量
4. **TileOffsets_median_cpu**: 中块级别的累积偏移
5. **TileOffsets_global_cpu**: 大块级别的累积偏移

## 计算执行

### SpMM内核调用

```cpp
cudaError_t result = SpMM_SplitK_API_bitmap_v3(
    stream,                     // CUDA流
    A_gpu,                      // 原始稠密矩阵A (用于某些计算)
    Compressed_Val_gpu,         // 压缩值数组
    TileOffsets_global_gpu,     // 大块偏移
    TileOffsets_median_gpu,     // 中块偏移
    bitmap_gpu,                 // bitmap数组
    max_nnz_intile_gpu,         // 最大块内非零元素数
    B_gpu,                      // 稠密矩阵B (K×N, column-major)
    C_gpu,                      // 输出矩阵C (M×N, column-major)
    M_GLOBAL, N_GLOBAL, K_GLOBAL,
    Reduction_Workspace,        // Split-K归约工作空间
    Split_K                     // Split-K分割数
);
```

### 内存布局要求

- **矩阵A**: Row-major布局 (M×K)
- **矩阵B**: Column-major布局 (K×N)
- **矩阵C**: Column-major布局 (M×N)

## 完整示例

### 编译和运行

```bash
# 1. 确保SpMM库已编译
cd build && make -j && cd ..

# 2. 编译自定义示例
make -f Makefile_custom

# 3. 运行示例
./custom_spmm_example 1024 4096 256 0.8
# 参数: M K N sparsity_ratio
```

### 示例输出

```
=== SpInfer 自定义数据处理示例 ===
矩阵维度: A(1024 x 4096) × B(4096 x 256) = C(1024 x 256)

步骤1: 准备输入数据
生成自定义稀疏矩阵: M=1024, K=4096, 稀疏度=0.80

步骤2: 稀疏格式转换
转换完成: 16个全局块, 838656个非零元素, 最大块内非零元素: 52416

步骤3: 数据传输到GPU
数据传输完成

步骤4: 执行SpMM计算
计算完成!
执行时间: 2.345 ms
性能: 456.78 TFLOPS

步骤5: 结果验证
结果矩阵中非零元素数量: 262144 / 262144

清理资源
程序执行完成!
```

## 性能优化建议

### 1. 矩阵维度选择

- **M维度**: 建议为64的倍数，以充分利用大块(64×64)
- **K维度**: 建议为64的倍数，以对齐tile边界
- **N维度**: 支持8, 16, 32, 64, 128等，会自动选择最优配置

### 2. 稀疏度设置

- **推荐范围**: 70%-95%稀疏度
- **过低稀疏度**: <50%时，稠密计算可能更高效
- **过高稀疏度**: >98%时，可能导致负载不均衡

### 3. Split-K策略

```cpp
// 根据矩阵大小选择Split-K
int optimal_split_k = 1;
if (K_GLOBAL > 8192) {
    optimal_split_k = 2;
}
if (K_GLOBAL > 16384) {
    optimal_split_k = 4;
}
```

### 4. 内存对齐

```cpp
// 确保内存对齐以获得最佳性能
size_t alignment = 128;  // 128字节对齐
half* aligned_A = (half*)aligned_alloc(alignment, sizeof(half) * M * K);
```

### 5. 批处理优化

对于多个矩阵乘法，可以重用转换后的稀疏格式：

```cpp
// 一次转换，多次使用
for (int batch = 0; batch < num_batches; batch++) {
    // 只需要更新B矩阵
    cudaMemcpy(B_gpu, B_batch[batch], sizeof(half) * K * N, cudaMemcpyHostToDevice);
    
    // 执行计算
    SpMM_SplitK_API_bitmap_v3(/* ... */);
}
```

## 故障排除

### 常见问题

1. **编译错误**: 确保CUDA版本 >= 12.0，GPU架构 >= sm_80
2. **内存不足**: 减少矩阵大小或增加GPU内存
3. **性能不佳**: 检查矩阵维度对齐和稀疏度设置
4. **结果错误**: 验证输入数据格式和内存布局

### 调试技巧

```cpp
// 启用调试模式
#define DEBUG_MODE
// 这将输出详细的转换信息

// 检查CUDA错误
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)
```

## 总结

通过本指南，您可以：

1. 使用多种方式生成或加载自定义稀疏矩阵数据
2. 理解SpInfer的分层稀疏格式转换过程
3. 正确调用SpMM内核进行高性能计算
4. 优化性能并解决常见问题

SpInfer的bitmap压缩格式特别适合处理不规则稀疏模式，在LLM推理等场景中能够显著提升性能。
