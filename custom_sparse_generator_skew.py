#!/usr/bin/env python3
"""
偏斜稀疏度分布生成器
按16x16块粒度控制稀疏度分布，生成偏斜分布的稀疏矩阵
前90%行使用高稀疏度，后10%行使用低稀疏度（稠密）
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

def generate_skewed_sparse_matrix(M: int, K: int, sparsity_dist: Dict[int, float], 
                                 block_size: int = 16, seed: int = 42) -> Tuple[np.ndarray, List[Dict]]:
    """
    生成偏斜稀疏度分布的矩阵
    前90%行使用高稀疏度，后10%行使用低稀疏度
    
    Args:
        M: 矩阵行数
        K: 矩阵列数  
        sparsity_dist: 稀疏度分布字典 {稀疏度: 比例}，如 {90: 0.9, 10: 0.1}
        block_size: 块大小，默认16
        seed: 随机种子
        
    Returns:
        matrix: 生成的偏斜稀疏矩阵
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
    
    # 按照比例分配稀疏度
    sparsity_configs = []
    for sparsity_value, ratio in sparsity_dist.items():
        sparsity_configs.append((sparsity_value, ratio))

    # 按比例排序（比例从大到小）
    sparsity_configs.sort(key=lambda x: x[1], reverse=True)

    print(f"稀疏度配置: {sparsity_configs}")

    # 计算每个稀疏度对应的行块数量
    sparsity_boundaries = []
    current_boundary = 0

    for sparsity_value, ratio in sparsity_configs:
        num_blocks_for_this_sparsity = int(num_blocks_M * ratio)
        start_block = current_boundary
        end_block = current_boundary + num_blocks_for_this_sparsity

        sparsity_boundaries.append({
            'sparsity': sparsity_value,
            'ratio': ratio,
            'start_block': start_block,
            'end_block': min(end_block, num_blocks_M),  # 确保不超出范围
            'num_blocks': min(end_block, num_blocks_M) - start_block
        })

        current_boundary = end_block

        print(f"稀疏度{sparsity_value}%: 第{start_block}-{min(end_block, num_blocks_M)-1}行块 "
              f"({start_block*block_size}-{min(end_block, num_blocks_M)*block_size-1}行), "
              f"比例{ratio*100:.1f}%")

    # 处理剩余的行块（如果有的话）
    if current_boundary < num_blocks_M:
        # 将剩余的行块分配给最后一个稀疏度
        sparsity_boundaries[-1]['end_block'] = num_blocks_M
        sparsity_boundaries[-1]['num_blocks'] = num_blocks_M - sparsity_boundaries[-1]['start_block']
        print(f"剩余行块分配给稀疏度{sparsity_boundaries[-1]['sparsity']}%")
    
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
            
            # 根据行位置确定稀疏度
            target_sparsity = None
            region_name = None

            # 查找当前行块属于哪个稀疏度区域
            for boundary in sparsity_boundaries:
                if boundary['start_block'] <= i < boundary['end_block']:
                    target_sparsity = boundary['sparsity']
                    region_name = f"sparsity_{boundary['sparsity']}"
                    break

            # 如果没有找到匹配的区域，使用默认值
            if target_sparsity is None:
                target_sparsity = sparsity_configs[0][0]  # 使用第一个稀疏度
                region_name = f"default_sparsity_{target_sparsity}"
            
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
                'total': total_elements,
                'region': region_name
            })
            
            block_idx += 1
    
    print(f"矩阵生成完成，总非零元素: {np.count_nonzero(matrix)}")
    
    # 统计各个稀疏度区域的信息
    for boundary in sparsity_boundaries:
        region_blocks = [info for info in block_info if info['region'] == f"sparsity_{boundary['sparsity']}"]
        if region_blocks:
            avg_sparsity = np.mean([b['actual_sparsity'] for b in region_blocks])
            print(f"稀疏度{boundary['sparsity']}%区域: {len(region_blocks)}个块, 平均稀疏度: {avg_sparsity:.2f}%")
    
    return matrix, block_info

def plot_sparsity_distribution(block_info: List[Dict], save_path: str = None):
    """
    绘制稀疏度分布直方图，区分不同稀疏度区域
    """
    # 获取所有不同的区域
    regions = list(set([info['region'] for info in block_info]))
    regions.sort()  # 排序以保持一致性

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 创建直方图
    bins = np.linspace(0, 100, 21)  # 20个区间

    # 为每个区域选择不同的颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # 绘制每个区域的直方图
    for i, region in enumerate(regions):
        region_sparsities = [info['actual_sparsity'] for info in block_info if info['region'] == region]
        if region_sparsities:
            color = colors[i % len(colors)]
            ax.hist(region_sparsities, bins=bins, alpha=0.7,
                    edgecolor='black', color=color,
                    label=f'{region} ({len(region_sparsities)} blocks)')

    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Number of Blocks', fontsize=12)
    ax.set_title(f'Skewed Block Sparsity Distribution (16x16 blocks)\n'
                 f'Total Blocks: {len(block_info)}', fontsize=14, fontweight='bold')

    # 添加图例
    ax.legend()

    # 添加统计信息
    stats_text = ""
    for region in regions:
        region_sparsities = [info['actual_sparsity'] for info in block_info if info['region'] == region]
        if region_sparsities:
            stats_text += f'{region}:\n'
            stats_text += f'  Mean: {np.mean(region_sparsities):.2f}%\n'
            stats_text += f'  Std: {np.std(region_sparsities):.2f}%\n'

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
    绘制偏斜稀疏模式图（采样显示）
    """
    M, K = matrix.shape

    # 采样显示（避免图像过大）
    if sample_ratio < 1.0:
        sample_M = int(M * sample_ratio)
        sample_K = int(K * sample_ratio)

        # 采样区域包含分界线，显示偏斜效果
        boundary_row = int(M * 0.9)
        start_m = max(0, boundary_row - sample_M // 2)
        start_m = min(start_m, M - sample_M)
        start_k = np.random.randint(0, K - sample_K + 1)

        sample_matrix = matrix[start_m:start_m+sample_M, start_k:start_k+sample_K]
        title_suffix = f" (Sampled around boundary: {sample_M}x{sample_K})"
    else:
        sample_matrix = matrix
        title_suffix = f" (Full Matrix: {M}x{K})"

    # 创建二值化模式图
    binary_pattern = (sample_matrix != 0).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制热力图
    im = ax.imshow(binary_pattern, cmap='Blues', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)

    ax.set_title(f'Skewed Sparse Matrix Pattern{title_suffix}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Input Dimension', fontsize=12)
    ax.set_ylabel('Output Dimension', fontsize=12)

    # 添加分界线标记（如果显示了分界线区域）
    if sample_ratio < 1.0:
        boundary_in_sample = int(sample_matrix.shape[0] * 0.5)  # 大概在中间
        ax.axhline(y=boundary_in_sample, color='red', linestyle='--', linewidth=2,
                  label='High/Low Sparse Boundary')
        ax.legend()

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
    output_dir = f"Input_date/skew_sparse_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 定义稀疏度分布
    sparsity_distribution = {
        40: 0.9,  
        90: 0.1, 
    }

    # 生成偏斜矩阵
    matrix, block_info = generate_skewed_sparse_matrix(
        M=16000, K=8192,
        sparsity_dist=sparsity_distribution,
        seed=42
    )

    # 可视化
    plot_sparsity_distribution(block_info, f"{output_dir}/skew_sparsity_distribution.pdf")
    plot_sparse_pattern(matrix, sample_ratio=1, save_path=f"{output_dir}/skew_sparse_pattern.pdf")

    # 保存矩阵
    save_matrix(matrix, f"{output_dir}/skew_sparse_matrix")

    # 创建latest符号链接，方便C++代码调用
    latest_link = "Input_date/skew_sparse_data_latest"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.basename(output_dir), latest_link)

    print(f"\n偏斜矩阵生成完成！文件保存在: {output_dir}")
    print(f"latest链接: {latest_link} -> {output_dir}")

    # 输出统计信息
    regions = list(set([info['region'] for info in block_info]))
    regions.sort()

    print(f"\n=== 统计信息 ===")
    print(f"总块数: {len(block_info)}")

    for region in regions:
        region_blocks = [info for info in block_info if info['region'] == region]
        if region_blocks:
            percentage = len(region_blocks) / len(block_info) * 100
            avg_sparsity = np.mean([b['actual_sparsity'] for b in region_blocks])
            print(f"{region}: {len(region_blocks)}个块 ({percentage:.1f}%), 平均稀疏度: {avg_sparsity:.2f}%")

    plt.show()
