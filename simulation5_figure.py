import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # 用于精确控制子图尺寸
from typing import Dict, List, Tuple

# -------------------------- 全局配置 --------------------------
DATA_FILE_PATH = r"D:\项目py\测试\simulation_results\sivr_real_network_results_20260605_110320.txt"
SAVE_DIR = "comparison_results"
DPI = 300
# 子图尺寸配置（视觉正方形，宽高一致）
SUBPLOT_WIDTH = 5  # 子图宽度（英寸）
SUBPLOT_HEIGHT = 5  # 子图高度（英寸，与宽度相等）
# 总画布尺寸：3列×子图宽，2行×子图高，预留边距避免标签截断
FIG_SIZE = (SUBPLOT_WIDTH * 3 + 0.5, SUBPLOT_HEIGHT * 2 + 0.5)
# 更新后的状态曲线样式
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
NUM_MARKER_POINTS = 15  # 曲线上标记点数量
# 子图下方标签列表
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

    # 用GridSpec精确控制子图位置和尺寸
    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(
        nrows=2, ncols=3,
        figure=fig,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        hspace=0.2,
        wspace=0.2
    )

    # 分配子图位置
    axes = [
        fig.add_subplot(gs[0, 0]),  # (a)
        fig.add_subplot(gs[0, 1]),  # (b)
        fig.add_subplot(gs[0, 2]),  # (c)
        fig.add_subplot(gs[1, 0]),  # (d)
        fig.add_subplot(gs[1, 1]),  # (e)
        fig.add_subplot(gs[1, 2])  # (f)
    ]

    fig.suptitle('', fontsize=18, y=0.99)

    # 绘制前3个子图：β参数组合实验
    beta_exps = all_results['beta_experiments']
    for idx in range(3):
        if idx >= len(beta_exps):
            break
        exp = beta_exps[idx]
        ax = axes[idx]
        subplot_label = SUBPLOT_LABELS[idx]

        # 绘制S/I/V/R四条曲线 - 使用单次plot调用同时绘制线条和标记点
        for state in ['S', 'I', 'Q', 'R']:
            data = exp[state]
            style = STATE_STYLES[state]

            # 创建包含所有数据点的数组
            x_data = np.arange(max_steps)
            y_data = np.array(data)

            # 单次plot调用，同时绘制线条和标记点
            ax.plot(
                x_data,
                y_data,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                marker=style['marker'],
                markersize=style['markersize'],
                markevery=marker_indices,  # 只在指定位置显示标记点
                label=style['label'],
                zorder=2
            )

        # 子图基础配置
        ax.set_title(' ', fontsize=14, pad=15)
        ax.set_xlabel('Time', fontsize=14, labelpad=3)
        ax.set_ylabel('Density', fontsize=14, labelpad=8)
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=10)

        # 子图下方添加标签
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

        # 绘制曲线 - 使用单次plot调用同时绘制线条和标记点
        for state in ['S', 'I', 'Q', 'R']:
            data = exp[state]
            style = STATE_STYLES[state]

            # 创建包含所有数据点的数组
            x_data = np.arange(max_steps)
            y_data = np.array(data)

            # 单次plot调用，同时绘制线条和标记点
            ax.plot(
                x_data,
                y_data,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                marker=style['marker'],
                markersize=style['markersize'],
                markevery=marker_indices,  # 只在指定位置显示标记点
                label=style['label'],
                zorder=2
            )

        # 子图基础配置
        ax.set_title('', fontsize=14, pad=15)
        ax.set_xlabel('Time', fontsize=14, labelpad=3)
        ax.set_ylabel('Density', fontsize=14, labelpad=8)
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=10)

        # 子图下方添加标签
        ax.text(
            x=0.5, y=-0.14,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    # 调整布局
    plt.tight_layout(rect=[0.02, 0.15, 0.98, 0.92])

    # 获取时间戳用于文件名
    file_name = os.path.basename(DATA_FILE_PATH)
    timestamp = file_name.replace('sivr_real_network_results_', '').replace('.txt', '')

    # 保存多种格式的图片
    formats = ['png', 'pdf', 'svg']  # 添加矢量图格式

    saved_files = []
    for fmt in formats:
        save_path = os.path.join(SAVE_DIR, f'combined_comparison_{timestamp}.{fmt}')
        if fmt == 'png':
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        else:
            # 矢量图格式不需要设置DPI
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        saved_files.append(save_path)
        print(f"{fmt.upper()}格式图表已保存至：{save_path}")

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

    print("正在生成综合对比图（两行三列，正方形子图+下方标签）...")
    saved_files = plot_sivr_results(all_results, max_steps)
    print("图表生成完成！")
    print("\n生成的矢量图文件：")
    for file in saved_files:
        if file.endswith(('.pdf', '.svg')):
            print(f"  - {file}")