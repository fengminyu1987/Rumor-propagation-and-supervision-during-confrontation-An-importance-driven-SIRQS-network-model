import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple

# -------------------------- 全局配置 --------------------------
DATA_FILE_PATH = r"D:\项目py\测试\simulation_results\sivr_all_results_20260605_013249.txt"
SAVE_DIR = "comparison_results"
# 矢量图格式配置（取消DPI设置，因为矢量图不依赖像素）
VECTOR_FORMATS = ['.pdf', '.svg']  # 可选的矢量图格式
# 子图尺寸配置
SUBPLOT_WIDTH = 5
SUBPLOT_HEIGHT = 5
FIG_SIZE = (SUBPLOT_WIDTH * 3 + 0.5, SUBPLOT_HEIGHT * 2 + 0.5)

# 状态曲线样式（保持不变）
STATE_STYLES = {
    'S': {
        'label': 'Ignorant (S)',
        'color': '#2E5EAA',
        'linestyle': '-',
        'marker': 'o',
        'linewidth': 2.5,
        'markersize': 8
    },
    'I': {
        'label': 'Spreader (I)',
        'color': '#C44E52',
        'linestyle': '-',
        'marker': 's',
        'linewidth': 2.5,
        'markersize': 8
    },
    'Q': {
        'label': 'Vigilant (Q)',
        'color': '#55A868',
        'linestyle': '-',
        'marker': '^',
        'linewidth': 2.5,
        'markersize': 8
    },
    'R': {
        'label': 'Stifler (R)',
        'color': '#8C6BB1',
        'linestyle': '-',
        'marker': 'd',
        'linewidth': 2.5,
        'markersize': 8
    }
}
NUM_MARKER_POINTS = 15
SUBPLOT_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


# -------------------------- 数据读取工具函数 --------------------------
def parse_sivr_txt(file_path: str) -> Tuple[Dict[str, List[Dict]], int]:
    all_results = {'beta_experiments': [], 'regulator_experiments': []}
    current_section = None
    current_exp = None
    max_steps = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if "第一部分: 不同β参数组合实验结果" in line:
            current_section = 'beta_experiments'
            continue
        if "第二部分: 不同监管者数量实验结果" in line:
            current_section = 'regulator_experiments'
            continue
        if line.startswith('--- '):
            if current_section is None:
                continue
            exp_label = line.strip('--- ').strip()
            current_exp = {'label': exp_label, 'params': {}, 'S': [], 'I': [], 'Q': [], 'R': []}
            all_results[current_section].append(current_exp)
            continue
        if current_exp and line.startswith('  ') and (':' in line):
            param_key, param_val = line.strip().split(':', 1)
            param_key = param_key.strip()
            param_val = param_val.strip()
            if param_key in ['β1值', 'β2值', '监管者数量']:
                param_val = float(param_val) if '.' in param_val else int(param_val)
            current_exp['params'][param_key] = param_val
            continue
        if current_exp and re.match(r'^\d+\t[\d\.]+\t[\d\.]+\t[\d\.]+\t[\d\.]+$', line):
            parts = line.split('\t')
            step = int(parts[0])
            current_exp['S'].append(float(parts[1]))
            current_exp['I'].append(float(parts[2]))
            current_exp['Q'].append(float(parts[3]))
            current_exp['R'].append(float(parts[4]))
            max_steps = max(max_steps, step + 1)

    return all_results, max_steps


def get_marker_indices(total_steps: int, num_points: int) -> List[int]:
    if num_points >= total_steps:
        return list(range(total_steps))
    indices = np.linspace(0, total_steps - 1, num_points, dtype=int)
    indices[0] = 0
    indices[-1] = total_steps - 1
    return sorted(list(set(indices)))


# -------------------------- 绘图主函数 --------------------------
def plot_sivr_results(all_results: Dict[str, List[Dict]], max_steps: int):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    marker_indices = get_marker_indices(max_steps, NUM_MARKER_POINTS)

    # 设置matplotlib以生成高质量矢量图
    plt.rcParams['pdf.fonttype'] = 42  # 确保文字在PDF中可编辑
    plt.rcParams['ps.fonttype'] = 42  # 确保文字在PostScript中可编辑
    plt.rcParams['svg.fonttype'] = 'none'  # 在SVG中保存文字为文字对象

    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(
        nrows=2, ncols=3,
        figure=fig,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        hspace=0.2,
        wspace=0.2
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2])
    ]

    # 绘制前3个子图：β参数组合实验
    beta_exps = all_results['beta_experiments']
    for idx in range(3):
        if idx >= len(beta_exps):
            break
        exp = beta_exps[idx]
        ax = axes[idx]
        subplot_label = SUBPLOT_LABELS[idx]

        for state in ['S', 'I', 'Q', 'R']:
            data = exp[state]
            style = STATE_STYLES[state]
            x_data = np.arange(max_steps)
            y_data = np.array(data)

            ax.plot(
                x_data,
                y_data,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                marker=style['marker'],
                markersize=style['markersize'],
                markevery=marker_indices,
                label=style['label'],
                zorder=2
            )

        ax.set_xlabel('Time', fontsize=14, labelpad=3)
        ax.set_ylabel('Density', fontsize=14, labelpad=8)
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=10)

        ax.text(
            x=0.5, y=-0.14,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    # 绘制后3个子图：监管者数量实验
    regulator_exps = all_results['regulator_experiments']
    for idx in range(3):
        exp_idx = idx
        ax_idx = idx + 3
        if exp_idx >= len(regulator_exps) or ax_idx >= len(axes):
            break
        exp = regulator_exps[exp_idx]
        ax = axes[ax_idx]
        subplot_label = SUBPLOT_LABELS[ax_idx]

        for state in ['S', 'I', 'Q', 'R']:
            data = exp[state]
            style = STATE_STYLES[state]
            x_data = np.arange(max_steps)
            y_data = np.array(data)

            ax.plot(
                x_data,
                y_data,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                marker=style['marker'],
                markersize=style['markersize'],
                markevery=marker_indices,
                label=style['label'],
                zorder=2
            )

        ax.set_xlabel('Time', fontsize=14, labelpad=3)
        ax.set_ylabel('Density', fontsize=14, labelpad=8)
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=10)

        ax.text(
            x=0.5, y=-0.14,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    plt.tight_layout(rect=[0.02, 0.15, 0.98, 0.92])

    # 保存多种矢量图格式
    file_name = os.path.basename(DATA_FILE_PATH)
    timestamp = file_name.replace('sivr_all_results_', '').replace('.txt', '')

    saved_files = []
    for format_ext in VECTOR_FORMATS:
        save_path = os.path.join(SAVE_DIR, f'combined_comparison_{timestamp}{format_ext}')
        plt.savefig(save_path, bbox_inches='tight', facecolor='white',
                    dpi=300, format=format_ext[1:])  # format_ext[1:] 去掉点号
        saved_files.append(save_path)
        print(f"矢量图已保存至：{save_path}")

    # 可选：同时保存PNG格式用于预览
    png_save_path = os.path.join(SAVE_DIR, f'combined_comparison_{timestamp}.png')
    plt.savefig(png_save_path, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(png_save_path)
    print(f"预览图已保存至：{png_save_path}")

    plt.show()
    return saved_files


# -------------------------- 主执行逻辑 --------------------------
if __name__ == "__main__":
    print(f"正在读取数据文件：{DATA_FILE_PATH}")
    try:
        all_results, max_steps = parse_sivr_txt(DATA_FILE_PATH)
    except Exception as e:
        print(f"数据读取失败：{str(e)}")
        exit(1)

    beta_count = len(all_results['beta_experiments'])
    regulator_count = len(all_results['regulator_experiments'])
    print(f"成功解析：β参数实验{beta_count}个，监管者数量实验{regulator_count}个，总时间步数{max_steps}")
    if beta_count < 3 or regulator_count < 3:
        print("警告：实验数量不足3个，部分子图可能为空")
        exit(1)

    print("正在生成高质量矢量图...")
    saved_files = plot_sivr_results(all_results, max_steps)
    print("图表生成完成！")
    print(f"生成的文件：{saved_files}")