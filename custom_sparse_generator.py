#!/usr/bin/env python3
"""
自定义稀疏度分布生成器
按16x16块粒度控制稀疏度分布，生成稀疏矩阵并可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, Tuple, List
import os
from datetime import datetime

# 设置高质量显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def generate_custom_sparse_matrix(M: int, K: int, sparsity_dist: Dict[int, float], 
                                 block_size: int = 16, seed: int = 42) -> Tuple[np.ndarray, List[Dict]]:
    """
    生成自定义稀疏度分布的矩阵
    
    Args:
        M: 矩阵行数
        K: 矩阵列数  
        sparsity_dist: 稀疏度分布字典 {稀疏度: 比例}，如 {70: 0.6, 20: 0.1}
        block_size: 块大小，默认16
        seed: 随机种子
        
    Returns:
        matrix: 生成的稀疏矩阵
        block_info: 块信息列表
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保矩阵维度是block_size的倍数
    M_aligned = (M // block_size) * block_size
    K_aligned = (K // block_size) * block_size
    
    print(f"原始矩阵大小: {M}x{K}")
    print(f"对齐后大小: {M_aligned}x{K_aligned}")
    
    # 计算块数量
    num_blocks_M = M_aligned // block_size
    num_blocks_K = K_aligned // block_size
    total_blocks = num_blocks_M * num_blocks_K
    
    print(f"块数量: {num_blocks_M} x {num_blocks_K} = {total_blocks}")
    
    # 验证分布比例总和
    total_ratio = sum(sparsity_dist.values())
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 分布比例总和为 {total_ratio:.3f}，将自动归一化")
        sparsity_dist = {k: v/total_ratio for k, v in sparsity_dist.items()}
    
    # 生成块的稀疏度分配
    block_sparsities = []
    for sparsity, ratio in sparsity_dist.items():
        num_blocks = int(total_blocks * ratio)
        block_sparsities.extend([sparsity] * num_blocks)
    
    # 处理由于舍入导致的块数量不足
    while len(block_sparsities) < total_blocks:
        # 添加最常见的稀疏度
        most_common_sparsity = max(sparsity_dist.keys(), key=lambda k: sparsity_dist[k])
        block_sparsities.append(most_common_sparsity)
    
    # 随机打乱块的稀疏度分配
    random.shuffle(block_sparsities)
    
    print(f"实际分配的块数量: {len(block_sparsities)}")
    
    # 初始化矩阵
    matrix = np.zeros((M_aligned, K_aligned), dtype=np.float16)
    block_info = []
    
    # 为每个块生成数据
    block_idx = 0
    for i in range(num_blocks_M):
        for j in range(num_blocks_K):
            # 块的位置
            start_row = i * block_size
            end_row = start_row + block_size
            start_col = j * block_size
            end_col = start_col + block_size
            
            # 获取该块的目标稀疏度
            target_sparsity = block_sparsities[block_idx]
            
            # 生成块数据
            block = np.random.randn(block_size, block_size).astype(np.float16)
            
            # 应用稀疏度
            total_elements = block_size * block_size
            num_zeros = int(total_elements * target_sparsity / 100)
            
            # 随机选择位置设为0
            flat_indices = np.random.choice(total_elements, num_zeros, replace=False)
            row_indices = flat_indices // block_size
            col_indices = flat_indices % block_size
            block[row_indices, col_indices] = 0
            
            # 将块放入矩阵
            matrix[start_row:end_row, start_col:end_col] = block
            
            # 记录块信息
            nnz = np.count_nonzero(block)
            actual_sparsity = (total_elements - nnz) / total_elements * 100
            
            block_info.append({
                'index': block_idx,
                'position': (i, j),
                'global_position': (start_row, start_col),
                'target_sparsity': target_sparsity,
                'actual_sparsity': actual_sparsity,
                'nnz': nnz,
                'total': total_elements
            })
            
            block_idx += 1
    
    print(f"矩阵生成完成，总非零元素: {np.count_nonzero(matrix)}")
    return matrix, block_info

def plot_sparsity_distribution(block_info: List[Dict], save_path: str = None):
    """
    绘制稀疏度分布直方图
    """
    sparsities = [info['actual_sparsity'] for info in block_info]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 创建直方图
    bins = np.linspace(0, 100, 21)  # 20个区间
    counts, bin_edges, _ = ax.hist(sparsities, bins=bins, alpha=0.7, 
                                  edgecolor='black', color='skyblue')
    
    # 计算百分比
    total_blocks = len(sparsities)
    percentages = (counts / total_blocks) * 100
    
    # 添加百分比标签
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        if count > 0:
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            ax.text(bin_center, count + total_blocks * 0.01, f'{percentage:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Number of Blocks', fontsize=12)
    ax.set_title(f'Block Sparsity Distribution (16x16 blocks)\n'
                 f'Total Blocks: {total_blocks}', fontsize=14, fontweight='bold')
    
    # 添加统计信息
    stats_text = f'Mean: {np.mean(sparsities):.2f}%\n'
    stats_text += f'Std: {np.std(sparsities):.2f}%\n'
    stats_text += f'Min: {np.min(sparsities):.2f}%\n'
    stats_text += f'Max: {np.max(sparsities):.2f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图保存至: {save_path}")
    
    return fig

def plot_sparse_pattern(matrix: np.ndarray, sample_ratio: float = 0.1, 
                       save_path: str = None):
    """
    绘制稀疏模式图（采样显示）
    """
    M, K = matrix.shape
    
    # 采样显示（避免图像过大）
    if sample_ratio < 1.0:
        sample_M = int(M * sample_ratio)
        sample_K = int(K * sample_ratio)
        
        # 随机采样区域
        start_m = np.random.randint(0, M - sample_M + 1)
        start_k = np.random.randint(0, K - sample_K + 1)
        
        sample_matrix = matrix[start_m:start_m+sample_M, start_k:start_k+sample_K]
        title_suffix = f" (Sampled {sample_ratio*100:.1f}%: {sample_M}x{sample_K})"
    else:
        sample_matrix = matrix
        title_suffix = f" (Full Matrix: {M}x{K})"
    
    # 创建二值化模式图
    binary_pattern = (sample_matrix != 0).astype(float)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制热力图
    im = ax.imshow(binary_pattern, cmap='Blues', aspect='auto', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_title(f'Sparse Matrix Pattern{title_suffix}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Input Dimension', fontsize=12)
    ax.set_ylabel('Output Dimension', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Non-zero Elements', fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Zero', 'Non-zero'])
    
    # 计算并显示稀疏度
    total_elements = sample_matrix.size
    nnz = np.count_nonzero(sample_matrix)
    sparsity = (total_elements - nnz) / total_elements * 100
    
    ax.text(0.02, 0.98, f'Sparsity: {sparsity:.2f}%\nNNZ: {nnz}\nTotal: {total_elements}',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模式图保存至: {save_path}")
    
    return fig

def save_matrix(matrix: np.ndarray, filename_base: str):
    """
    保存矩阵为多种格式
    """
    # 保存为numpy格式
    npy_path = f"{filename_base}.npy"
    np.save(npy_path, matrix)
    print(f"矩阵保存至: {npy_path}")
    
    # 保存为二进制格式（SpInfer兼容）
    bin_path = f"{filename_base}.bin"
    matrix.astype(np.float16).tofile(bin_path)
    print(f"二进制格式保存至: {bin_path}")
    
    return npy_path, bin_path

# 使用示例
if __name__ == "__main__":
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"Input_date/sparse_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 定义稀疏度分布
    sparsity_distribution = {
        90: 0.9,  # 70%稀疏度的块占60%
        10: 0.1,  # 20%稀疏度的块占10%
        # 90: 0.2,  # 90%稀疏度的块占20%
        # 50: 0.1   # 50%稀疏度的块占10%
    }

    # 生成矩阵
    matrix, block_info = generate_custom_sparse_matrix(
        M=16000, K=8192,
        sparsity_dist=sparsity_distribution,
        seed=42
    )

    # 可视化
    plot_sparsity_distribution(block_info, f"{output_dir}/sparsity_distribution.pdf")
    plot_sparse_pattern(matrix, sample_ratio=1, save_path=f"{output_dir}/sparse_pattern.pdf")

    # 保存矩阵
    save_matrix(matrix, f"{output_dir}/custom_sparse_matrix")

    # 创建latest符号链接，方便C++代码调用
    latest_link = "Input_date/sparse_data_latest"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.basename(output_dir), latest_link)

    print(f"\n生成完成！文件保存在: {output_dir}")
    print(f"latest链接: {latest_link} -> {output_dir}")
    plt.show()
