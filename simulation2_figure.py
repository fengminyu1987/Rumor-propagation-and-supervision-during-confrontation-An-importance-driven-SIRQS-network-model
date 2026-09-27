import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os
import re

# -------------------------- 全局样式配置 --------------------------
# 自定义色图
custom_colors = [
    (57 / 255, 81 / 255, 162 / 255),  # 深蓝色
    (114 / 255, 170 / 255, 207 / 255),  # 中蓝色
    (202 / 255, 232 / 255, 242 / 255),  # 浅蓝色
    (254 / 255, 251 / 255, 186 / 255),  # 浅黄色
    (253 / 255, 185 / 255, 107 / 255),  # 浅橙色
    (236 / 255, 93 / 255, 59 / 255),  # 橙色
    (168 / 255, 3 / 255, 38 / 255)  # 深红色
]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors, N=256)

# 全局绘图样式配置 - 优化矢量图输出
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'lines.markersize': 6,
    'lines.linewidth': 1.2,
    # 矢量图优化设置
    'svg.fonttype': 'none',  # 确保字体在SVG中可编辑
    'pdf.fonttype': 42,  # 确保字体在PDF中可编辑
    'ps.fonttype': 42,  # 确保字体在PS中可编辑
})

# 目标感染率
TARGET_RATE = 0.02


def parse_sivr_txt_data(txt_path):
    """解析SIVR模型数据格式"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 参数配置
    param_configs = [
        {
            'name': 'mu',
            'range': np.linspace(0.01, 0.5, 15),
            'label': 'μ (I→R康复率)',
            'latex': r'$\mu$'
        },
        {
            'name': 'gamma1',
            'range': np.linspace(0.01, 0.15, 15),
            'label': 'γ₁ (V→S免疫消退率)',
            'latex': r'$\gamma_1$'
        },
        {
            'name': 'gamma2',
            'range': np.linspace(0.01, 0.1, 15),
            'label': 'γ₂ (R→V免疫消退率)',
            'latex': r'$\gamma_2$'
        },
        {
            'name': 'xi',
            'range': np.linspace(0.1, 3.0, 15),
            'label': 'ξ (谣言对抗系数)',
            'latex': r'$\xi$'
        },
        {
            'name': 'w_p',
            'range': np.linspace(0.1, 0.9, 15),
            'label': 'w_p (战略权重参数)',
            'latex': r'$w_p$'
        },
        {
            'name': 'theta',
            'range': np.linspace(0.1, 0.5, 15),
            'label': 'θ (β₂/β₁比值)',
            'latex': r'$\theta$'
        }
    ]

    # 查找所有参数块
    param_pattern = r'参数\s*\d+:.*?(?=参数\s*\d+:|$)'
    param_blocks = re.findall(param_pattern, content, re.DOTALL)

    if not param_blocks:
        param_blocks = re.split(r'-{50,}', content)[1:7]

    all_results = []

    for block_idx, block in enumerate(param_blocks):
        print(f"正在解析参数块 {block_idx + 1}/6: {param_configs[block_idx]['label']}")

        # 提取参数范围
        param_range_match = re.search(r'参数范围[：:]\s*([\d\.]+)\s*[—-]\s*([\d\.]+)', block)
        if param_range_match:
            start_val = float(param_range_match.group(1))
            end_val = float(param_range_match.group(2))
            param_range = np.linspace(start_val, end_val, 8)
            print(f"  参数范围: {start_val} - {end_val} ({len(param_range)}个点)")
        else:
            print(f"  警告: 未找到参数范围，使用默认范围")
            param_range = np.linspace(0.01, 0.5, 8)

        # 根据子图位置确定β1范围
        if block_idx < 3:
            beta1_range_full = np.linspace(0.1, 0.8, 12)
        else:
            beta1_range_full = np.linspace(0.0, 0.3, 12)

        # 提取数据行
        lines = block.split('\n')
        steps_matrix = []
        found_beta1_values = []

        for line in lines:
            beta1_pattern = r'^\s*(\d+\.\d+)\s+'
            match = re.match(beta1_pattern, line)
            if match:
                beta_val = float(match.group(1))
                numbers = re.findall(r'[\d\.]+', line)
                if len(numbers) >= 9:
                    try:
                        step_vals = [float(x) for x in numbers[1:9]]
                        found_beta1_values.append(beta_val)
                        steps_matrix.append(step_vals)
                    except ValueError:
                        continue

        if not steps_matrix:
            print("  使用备用方法提取数据...")
            for line in lines:
                numbers = re.findall(r'[\d\.]+', line)
                if len(numbers) >= 5:
                    try:
                        first_num = float(numbers[0])
                        if ((block_idx < 3 and 0.1 <= first_num <= 0.8) or
                                (block_idx >= 3 and 0.0 <= first_num <= 0.3)):
                            if len(numbers) >= 9:
                                step_vals = [float(x) for x in numbers[1:9]]
                                if first_num not in found_beta1_values:
                                    found_beta1_values.append(first_num)
                                    steps_matrix.append(step_vals)
                    except ValueError:
                        continue

        if not steps_matrix:
            raise ValueError(f"参数{block_idx + 1}未找到有效的矩阵数据")

        steps_matrix = np.array(steps_matrix, dtype=np.float64)

        # 对找到的β1值进行排序
        sorted_indices = np.argsort(found_beta1_values)
        found_beta1_values = [found_beta1_values[i] for i in sorted_indices]
        steps_matrix = steps_matrix[sorted_indices]

        print(f"  成功解析: β1值数量={len(found_beta1_values)}, 参数值数量={len(param_range)}")

        all_results.append((steps_matrix, param_configs[block_idx], found_beta1_values, param_range))

    return all_results


def plot_heatmaps_no_contour(all_results):
    """绘制热力图并输出高质量矢量图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 找到所有结果中的最小和最大值，用于统一色标范围
    all_min = min(np.min(result[0]) for result in all_results)
    all_max = max(np.max(result[0]) for result in all_results)

    # 子图标签
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for idx, (steps_matrix, config, beta1_range, param_range) in enumerate(all_results):
        ax = axes[idx]

        # 平滑热力图数据
        smoothed_matrix = gaussian_filter(steps_matrix, sigma=1.0)

        # 计算坐标轴范围（归一化到[0,1]）
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1

        # 绘制热力图
        im = ax.imshow(
            smoothed_matrix,
            cmap=custom_cmap,
            aspect='auto',
            vmin=all_min,
            vmax=all_max,
            interpolation='spline16',
            origin='lower',
            extent=[x_min, x_max, y_min, y_max]
        )

        # 设置坐标轴标签和刻度
        x_ticks = np.linspace(0, 1, 5)
        x_tick_positions = np.linspace(0, len(param_range) - 1, 5)
        x_tick_labels = [f'{param_range[int(pos)]:.2f}' if param_range[int(pos)] < 1
                         else f'{param_range[int(pos)]:.1f}' for pos in x_tick_positions]

        y_ticks = np.linspace(0, 1, 5)
        y_tick_positions = np.linspace(0, len(beta1_range) - 1, 5)
        y_tick_labels = [f'{beta1_range[int(pos)]:.2f}' for pos in y_tick_positions]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        # 设置轴标签
        ax.set_xlabel(f"{config['latex']} ", fontsize=14, labelpad=1)
        ax.set_ylabel(r'$\beta$', fontsize=14, labelpad=2)

        # 添加子图标签
        ax.text(0.5, -0.185, subplot_labels[idx], transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='center', va='center')

    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Iterations Until Infection Rate ≤ {TARGET_RATE}', rotation=270, labelpad=20, fontsize=15)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # 创建输出目录
    if not os.path.exists("heatmap_results"):
        os.makedirs("heatmap_results")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存多种格式的矢量图
    output_files = []

    # 1. PDF格式（最常用的矢量图格式）
    pdf_file = f"heatmap_results/sivr_heatmaps_no_contour_{timestamp}.pdf"
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white',
                format='pdf', transparent=False)
    output_files.append(pdf_file)

    # 2. SVG格式（可编辑的矢量图格式）
    svg_file = f"heatmap_results/sivr_heatmaps_no_contour_{timestamp}.svg"
    plt.savefig(svg_file, dpi=300, bbox_inches='tight', facecolor='white',
                format='svg', transparent=False)
    output_files.append(svg_file)

    # 3. EPS格式（用于出版的高质量矢量图）
    eps_file = f"heatmap_results/sivr_heatmaps_no_contour_{timestamp}.eps"
    plt.savefig(eps_file, dpi=300, bbox_inches='tight', facecolor='white',
                format='eps', transparent=False)
    output_files.append(eps_file)

    # 4. 高分辨率PNG（用于预览）
    png_file = f"heatmap_results/sivr_heatmaps_no_contour_{timestamp}.png"
    plt.savefig(png_file, dpi=600, bbox_inches='tight', facecolor='white',
                format='png', transparent=False)
    output_files.append(png_file)

    print(f"\n图像已保存至 heatmap_results 文件夹:")
    for file in output_files:
        print(f"  - {os.path.basename(file)}")

    print(f"\n矢量图说明:")
    print(f"  - PDF格式: 适合LaTeX文档和一般使用")
    print(f"  - SVG格式: 可编辑，适合Inkscape等矢量图软件")
    print(f"  - EPS格式: 适合科学出版和高质量打印")
    print(f"  - PNG格式: 高分辨率位图，用于预览")

    plt.show()


def main():
    """主函数"""
    # 指定数据文件路径
    data_file = r"D:\项目py\测试\heatmap_results\sivr_heatmap_data11_20260605_022300.txt"

    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误：数据文件不存在 - {data_file}")
        print("请检查文件路径是否正确")
        return

    # 加载数据
    print("正在解析实验数据...")
    try:
        all_results = parse_sivr_txt_data(data_file)
        print("数据解析完成！")
    except Exception as e:
        print(f"数据解析错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 绘制热力图
    print("正在生成热力图和矢量图...")
    plot_heatmaps_no_contour(all_results)


if __name__ == "__main__":
    main()