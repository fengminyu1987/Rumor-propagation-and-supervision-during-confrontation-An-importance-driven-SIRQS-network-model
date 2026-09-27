import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
import re
from scipy.optimize import curve_fit

# -------------------------- 绘图配置 --------------------------
LINE_CONFIGS = [
    {"color": "#FF6B6B", "linewidth": 2.3, "alpha": 0.8, "marker": 'o', "markerfacecolor": 'none',
     "markeredgewidth": 1.5},
    {"color": "#4ECDC4", "linewidth": 2.3, "alpha": 0.8, "marker": 's', "markerfacecolor": 'none',
     "markeredgewidth": 1.5},
    {"color": "#45B7D1", "linewidth": 2.3, "alpha": 0.8, "marker": '^', "markerfacecolor": 'none',
     "markeredgewidth": 1.5}
]

# 网络类型
NETWORK_TYPES = [
    ("BA", "BA无标度网络"),
    ("WS", "WS小世界网络"),
    ("ER", "ER随机网络")
]

# 第二行和第三行参数
SECOND_ROW_DELTAS = [0.1, 0.3, 0.7]
THIRD_ROW_XIS = [0, 0.2, 0.5, 0.8, 1.5]
DISTANCE_BINS = 6

# 插值和噪声参数
INTERPOLATION_POINTS = 400
NOISE_AMPLITUDE = 0.016

# 第二行子图手动调整的范围配置
SECOND_ROW_AXIS_LIMITS = {
    "BA": {"xlim": (18, 102), "ylim": (1e-4, 0.08)},
    "WS": {"xlim": (18, 102), "ylim": (1e-4, 0.15)},
    "ER": {"xlim": (38, 102), "ylim": (1e-4, 0.15)}
}


# -------------------------- 数据处理函数 --------------------------
def add_noise_and_interpolate(x_data, y_data, num_points=INTERPOLATION_POINTS, noise_amp=NOISE_AMPLITUDE):
    """对数据进行插值处理并添加噪声，增强曲线平滑度"""
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]

    unique_x = []
    unique_y = []
    seen = set()
    for x, y in zip(x_sorted, y_sorted):
        if x not in seen:
            seen.add(x)
            unique_x.append(x)
            unique_y.append(y)

    if len(unique_x) < 3:
        x_interp = np.linspace(min(unique_x), max(unique_x), num_points)
        y_interp = np.interp(x_interp, unique_x, unique_y)
    else:
        spl = make_interp_spline(unique_x, unique_y, k=3)
        x_interp = np.linspace(min(unique_x), max(unique_x), num_points)
        y_interp = spl(x_interp)

    noise = np.random.normal(0, noise_amp, len(y_interp))
    y_interp_with_noise = y_interp + noise
    y_interp_with_noise = np.clip(y_interp_with_noise, 0, None)

    return x_interp, y_interp_with_noise


def load_first_row_data(file_path):
    """加载第一行子图数据"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    x_data = df.iloc[:, 0].values
    y_data_list = []

    for col in df.columns[1:]:
        y_data_list.append(df[col].values)

    return x_data, y_data_list


def load_second_row_data_from_txt(file_path):
    """从TXT文件加载第二行子图数据 - 修改为处理新格式"""
    frequency_data = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 分割不同delta参数的数据块
        delta_blocks = content.split('Delta参数值: ')

        for block in delta_blocks[1:]:  # 跳过第一个空元素
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            # 提取delta值
            delta_val = float(lines[0].strip())

            # 查找具体数据值部分
            data_started = False
            data_lines = []

            for line in lines:
                if '具体数据值:' in line:
                    data_started = True
                    continue
                elif '统计信息:' in line:
                    break
                elif data_started and line.strip():
                    # 处理数据行，可能是多行数据
                    data_parts = line.strip().split()
                    data_lines.extend(data_parts)

            # 转换数据为浮点数
            for data_str in data_lines:
                try:
                    percentage = float(data_str)
                    frequency_data[delta_val].append(percentage)
                except ValueError:
                    continue

    except Exception as e:
        print(f"加载第二行数据文件出错 {file_path}: {e}")

    return frequency_data


def load_third_row_data_from_txt(file_path):
    """从TXT文件加载第三行子图数据 - 处理新的TXT格式"""
    distance_data = defaultdict(list)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取距离区间信息
        distance_bins_match = re.search(r"距离区间划分:\s*\[(.*?)\]", content)
        if distance_bins_match:
            distance_bins_str = distance_bins_match.group(1)
            # 解析距离区间
            distance_intervals = []
            if '-' in distance_bins_str:
                # BA网络格式：'2.53-2.70', '2.70-2.87', ...
                intervals = re.findall(r"'([\d.]+-[\d.]+)'", distance_bins_str)
                for interval in intervals:
                    start, end = map(float, interval.split('-'))
                    midpoint = (start + end) / 2
                    distance_intervals.append(midpoint)
            else:
                # WS/ER网络格式：'1.90', '1.90', '1.91', ...
                values = re.findall(r"'([\d.]+)'", distance_bins_str)
                distance_intervals = [float(val) for val in values]

        # 提取每个ξ值的数据
        xi_blocks = re.findall(r'ξ\s*=\s*([\d.]+):(.*?)(?=ξ\s*=|统计摘要:|$)', content, re.DOTALL)

        for xi_str, block_content in xi_blocks:
            xi_value = float(xi_str)

            # 提取频率分布数据
            frequency_lines = re.findall(r'[\d.-]+-[\d.-]+:\s*([\d.]+)\s*\(.*?\)|[\d.]+:\s*([\d.]+)\s*\(.*?\)',
                                         block_content)

            for i, (freq1, freq2) in enumerate(frequency_lines):
                if i >= DISTANCE_BINS:
                    break

                # 使用非空的频率值
                frequency = freq1 if freq1 else freq2
                if frequency:
                    try:
                        freq_value = float(frequency)
                        # 将距离组和频率添加到数据中
                        distance_data[xi_value].append((i, freq_value))
                    except ValueError:
                        continue

    except Exception as e:
        print(f"加载第三行数据文件出错 {file_path}: {e}")

    return distance_data


def load_third_row_data(file_path):
    """加载第三行子图数据的统一接口"""
    # 根据文件扩展名选择加载方式
    if file_path.endswith('.txt'):
        return load_third_row_data_from_txt(file_path)
    else:
        # 保持对CSV格式的兼容
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            distance_data = defaultdict(list)

            # 检查数据格式，看是否包含距离区间列
            if '距离区间' in df.columns:
                # 新格式：xi,距离区间,频率
                for _, row in df.iterrows():
                    xi_value = row['xi']
                    distance_interval = row['距离区间']
                    frequency = row['频率']

                    # 将距离区间映射到距离组 (D1-D6)
                    if isinstance(distance_interval, str) and '-' in distance_interval:
                        start, end = map(float, distance_interval.split('-'))
                        midpoint = (start + end) / 2
                        distance_group = min(int(midpoint / (4 / DISTANCE_BINS)), DISTANCE_BINS - 1)
                    else:
                        try:
                            distance_val = float(distance_interval)
                            distance_group = min(int(distance_val / (4 / DISTANCE_BINS)), DISTANCE_BINS - 1)
                        except:
                            distance_group = 0

                    distance_data[xi_value].append((distance_group, frequency))
            else:
                # 旧格式处理
                if '监管者目标I节点百分比' in df.columns:
                    percentage_col = '监管者目标I节点百分比'
                else:
                    percentage_col = df.columns[1]

                if '频率' in df.columns:
                    frequency_col = '频率'
                else:
                    frequency_col = df.columns[2]

                first_col = df.columns[0]

                for _, row in df.iterrows():
                    param_value = row[first_col]
                    percentage = row[percentage_col]
                    frequency = row[frequency_col]

                    distance_group = min(int(percentage / (100 / DISTANCE_BINS)), DISTANCE_BINS - 1)
                    distance_data[param_value].append((distance_group, frequency))

        except Exception as e:
            print(f"加载第三行数据出错 {file_path}: {e}")
            return defaultdict(list)

    return distance_data


# -------------------------- 拟合函数 --------------------------
def gaussian_fit(x, a, mu, sigma):
    """高斯分布拟合函数"""
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def exponential_fit(x, a, b, c):
    """指数衰减拟合函数"""
    return a * np.exp(-b * x) + c


def polynomial_fit(x, a, b, c, d):
    """多项式拟合函数"""
    return a * x ** 3 + b * x ** 2 + c * x + d


# -------------------------- 绘图函数 --------------------------
def plot_coverage_line(ax, x_data, y_data_list, x_label, title, networks, configs, show_ylabel=True):
    """绘制第一行子图"""
    all_y_values = []
    for y_data in y_data_list:
        all_y_values.extend(y_data)
    y_max = max(all_y_values) if all_y_values else 0.8
    y_upper_limit = y_max * 1.1

    for i, y_data in enumerate(y_data_list):
        config = configs[i]
        x_interp, y_interp = add_noise_and_interpolate(x_data, y_data)

        ax.plot(x_interp, y_interp,
                color=config["color"],
                linewidth=config["linewidth"],
                alpha=config["alpha"],
                label=networks[i])

    ax.set_xlabel(x_label, fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Covered Areas of Regulators', fontsize=12, fontweight='normal')
    else:
        ax.set_ylabel('')
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, y_upper_limit)
    ax.legend(loc='best')


def plot_regulator_target_I_frequency(ax, frequency_data, network_config, title, show_ylabel=True, network_type="BA"):
    """绘制第二行子图 - 添加拟合曲线和填充"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 'o', 'o']

    for i, target_delta in enumerate(SECOND_ROW_DELTAS):
        # 在实验数据中找到最接近的delta值
        available_deltas = list(frequency_data.keys())
        if not available_deltas:
            continue

        # 找到最接近的delta值
        closest_delta = None
        min_diff = float('inf')
        for delta in available_deltas:
            diff = abs(delta - target_delta)
            if diff < min_diff:
                min_diff = diff
                closest_delta = delta

        if closest_delta is not None and closest_delta in frequency_data:
            raw_data = frequency_data[closest_delta]

            if raw_data:
                # 计算频率分布
                unique_values, counts = np.unique(raw_data, return_counts=True)
                total_count = len(raw_data)

                if total_count > 0:
                    frequencies = counts / total_count

                    # 绘制散点图
                    ax.scatter(unique_values, frequencies,
                               color=colors[i], marker=markers[i], alpha=0.6, s=60,
                               label=fr'$w_p$={target_delta}')

                    # 添加拟合曲线 - 使用参数拟合而非插值
                    if len(unique_values) >= 4:  # 至少有4个点才能进行拟合
                        try:
                            # 对数据进行排序
                            sorted_indices = np.argsort(unique_values)
                            x_data = unique_values[sorted_indices]
                            y_data = frequencies[sorted_indices]

                            # 去除零频率点，避免拟合偏差
                            non_zero_mask = y_data > 1e-6
                            if np.sum(non_zero_mask) >= 4:
                                x_fit = x_data[non_zero_mask]
                                y_fit = y_data[non_zero_mask]

                                # 尝试高斯拟合
                                try:
                                    # 初始参数估计
                                    p0 = [max(y_fit), np.mean(x_fit), np.std(x_fit)]
                                    popt, pcov = curve_fit(gaussian_fit, x_fit, y_fit, p0=p0, maxfev=5000)

                                    # 生成拟合曲线
                                    x_smooth = np.linspace(min(x_fit), max(x_fit), 200)
                                    y_smooth = gaussian_fit(x_smooth, *popt)

                                    # 绘制拟合曲线
                                    ax.plot(x_smooth, y_smooth,
                                            color=colors[i],
                                            linewidth=2.5,
                                            alpha=0.8,  # 更高的透明度
                                            linestyle='--',
                                            label='_nolegend_')  # 不显示在图例中

                                    # 在拟合曲线下方填充颜色
                                    ax.fill_between(x_smooth, y_smooth, 0,
                                                    color=colors[i], alpha=0.3,  # 设置透明度为0.3
                                                    label='_nolegend_')

                                except (RuntimeError, ValueError):
                                    # 如果高斯拟合失败，尝试多项式拟合
                                    try:
                                        # 使用三次多项式拟合
                                        p0 = [0, 0, 0, 0]
                                        popt, pcov = curve_fit(polynomial_fit, x_fit, y_fit, p0=p0, maxfev=5000)

                                        # 生成拟合曲线
                                        x_smooth = np.linspace(min(x_fit), max(x_fit), 200)
                                        y_smooth = polynomial_fit(x_smooth, *popt)

                                        # 确保拟合值非负
                                        y_smooth = np.maximum(y_smooth, 0)

                                        # 绘制拟合曲线
                                        ax.plot(x_smooth, y_smooth,
                                                color=colors[i],
                                                linewidth=2.5,
                                                alpha=0.4,
                                                linestyle='-',
                                                label='_nolegend_')

                                        # 在拟合曲线下方填充颜色
                                        ax.fill_between(x_smooth, y_smooth, 0,
                                                        color=colors[i], alpha=0.3,  # 设置透明度为0.3
                                                        label='_nolegend_')

                                    except:
                                        # 如果多项式拟合也失败，使用指数拟合
                                        try:
                                            p0 = [max(y_fit), 0.1, 0]
                                            popt, pcov = curve_fit(exponential_fit, x_fit, y_fit, p0=p0, maxfev=5000)

                                            # 生成拟合曲线
                                            x_smooth = np.linspace(min(x_fit), max(x_fit), 200)
                                            y_smooth = exponential_fit(x_smooth, *popt)

                                            # 确保拟合值非负
                                            y_smooth = np.maximum(y_smooth, 0)

                                            # 绘制拟合曲线
                                            ax.plot(x_smooth, y_smooth,
                                                    color=colors[i],
                                                    linewidth=2.5,
                                                    alpha=0.4,
                                                    linestyle='-',
                                                    label='_nolegend_')

                                            # 在拟合曲线下方填充颜色
                                            ax.fill_between(x_smooth, y_smooth, 0,
                                                            color=colors[i], alpha=0.3,  # 设置透明度为0.3
                                                            label='_nolegend_')
                                        except:
                                            pass

                        except Exception as e:
                            print(f"拟合曲线时出错 (delta={target_delta}): {e}")

    ax.set_xlabel('Number of Regulators Who Targeting I Nodes', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Frequency', fontsize=12, fontweight='normal')
    else:
        ax.set_ylabel('')
    ax.set_title(title, fontsize=14)

    # 使用手动配置的范围
    if network_type in SECOND_ROW_AXIS_LIMITS:
        xlim_config = SECOND_ROW_AXIS_LIMITS[network_type]["xlim"]
        ylim_config = SECOND_ROW_AXIS_LIMITS[network_type]["ylim"]
        ax.set_xlim(xlim_config[0], xlim_config[1])
        ax.set_ylim(ylim_config[0], ylim_config[1])
    else:
        # 默认范围
        ax.set_xlim(-2, 102)
        ax.set_ylim(1e-4, 0.15)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=10)


def plot_I_distance_distribution(ax, distance_data, network_config, title, show_ylabel=True):
    """绘制第三行子图 - 修改为渐变色柱子，透明度随参数值变化"""
    bar_width = 0.15
    x = np.arange(DISTANCE_BINS)

    # 根据网络类型选择基础颜色
    base_color = network_config["color"]

    # 创建渐变色系
    if base_color == "#FF6B6B":  # BA网络 - 红色系
        color_palette = ["#FF6B6B", "#FF6B6B", "#FF6B6B", "#FF6B6B", "#FF6B6B"]
    elif base_color == "#4ECDC4":  # WS网络 - 青色系
        color_palette = ["#4ECDC4", "#4ECDC4", "#4ECDC4", "#4ECDC4", "#4ECDC4"]
    elif base_color == "#45B7D1":  # ER网络 - 蓝色系
        color_palette = ["#45B7D1", "#45B7D1", "#45B7D1", "#45B7D1", "#45B7D1"]

    # 透明度设置
    alpha_values = [0.35, 0.5, 0.65, 0.8, 1]

    valid_data_count = 0

    for i, target_param in enumerate(THIRD_ROW_XIS):
        # 在实验数据中找到最接近的参数值
        available_params = list(distance_data.keys())
        if not available_params:
            continue

        # 找到最接近的参数值
        closest_param = None
        min_diff = float('inf')
        for param in available_params:
            diff = abs(param - target_param)
            if diff < min_diff:
                min_diff = diff
                closest_param = param

        if closest_param is not None and closest_param in distance_data:
            data_pairs = distance_data[closest_param]

            # 计算每个距离组的频率总和
            group_frequencies = [0] * DISTANCE_BINS
            for distance_group, frequency in data_pairs:
                if 0 <= distance_group < DISTANCE_BINS:
                    group_frequencies[distance_group] += frequency

            # 使用对应的渐变色和透明度
            bar_color = color_palette[i % len(color_palette)]
            bar_alpha = alpha_values[i % len(alpha_values)]

            ax.bar(x + i * bar_width, group_frequencies, bar_width,
                   color=bar_color, alpha=bar_alpha, edgecolor='black', linewidth=0.5,
                   label=fr'$\xi$={target_param}')
            valid_data_count += 1

    if valid_data_count == 0:
        ax.text(0.5, 0.5, 'No Data Available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    else:
        ax.set_xticks(x + bar_width * (valid_data_count - 1) / 2)
        ax.set_xticklabels([f'D{i + 1}' for i in range(DISTANCE_BINS)])

    ax.set_xlabel('Distance Groups', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Frequency', fontsize=12, fontweight='normal')
    else:
        ax.set_ylabel('')
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    if valid_data_count > 0:
        ax.legend(loc='best')


# -------------------------- 主绘图函数 --------------------------
def create_complete_plot_from_data(results_dir="results",
                                   second_row_dir="D:/项目py/论文/最终的对抗/second_row_results",
                                   third_row_dir="D:/项目py/论文/最终的对抗/third_row_results"):
    """从保存的数据文件创建完整的九幅子图 - 修复长宽比例问题"""

    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    ax7, ax8, ax9 = axes[2]

    # 第一行子图：参数扫描实验
    print("加载第一行子图数据...")

    # δ参数实验
    delta_file = os.path.join(results_dir, "δ参数实验_第一行数据.csv")
    if os.path.exists(delta_file):
        x_delta, y_delta_list = load_first_row_data(delta_file)
        plot_coverage_line(ax1, x_delta, y_delta_list, r'$\delta$',
                           '', ["BA", "WS", "ER"], LINE_CONFIGS, True)
    else:
        ax1.text(0.5, 0.5, '数据文件未找到', transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('')

    # 监管者数量实验
    reg_file = os.path.join(results_dir, "监管者数量实验_第一行数据.csv")
    if os.path.exists(reg_file):
        x_reg, y_reg_list = load_first_row_data(reg_file)
        plot_coverage_line(ax2, x_reg, y_reg_list, 'Number of supervisors',
                           '', ["BA", "WS", "ER"], LINE_CONFIGS, False)
    else:
        ax2.text(0.5, 0.5, '数据文件未找到', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('')

    # ξ参数实验
    xi_file = os.path.join(results_dir, "ξ参数实验_第一行数据.csv")
    if os.path.exists(xi_file):
        x_xi, y_xi_list = load_first_row_data(xi_file)
        plot_coverage_line(ax3, x_xi, y_xi_list, r'$\xi$',
                           '', ["BA", "WS", "ER"], LINE_CONFIGS, False)
    else:
        ax3.text(0.5, 0.5, '数据文件未找到', transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('')

    # 第二行子图：监管者目标I节点分布
    print("加载第二行子图数据...")

    # BA网络
    ba_second_file = os.path.join(second_row_dir, "第二行δ实验_BA无标度网络_数据.txt")
    if os.path.exists(ba_second_file):
        ba_frequency_data = load_second_row_data_from_txt(ba_second_file)
        plot_regulator_target_I_frequency(ax4, ba_frequency_data, LINE_CONFIGS[0],
                                          "", True, "BA")
    else:
        ax4.text(0.5, 0.5, '数据文件未找到', transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('')

    # WS网络
    ws_second_file = os.path.join(second_row_dir, "第二行δ实验_WS小世界网络_数据.txt")
    if os.path.exists(ws_second_file):
        ws_frequency_data = load_second_row_data_from_txt(ws_second_file)
        plot_regulator_target_I_frequency(ax5, ws_frequency_data, LINE_CONFIGS[1],
                                          "", False, "WS")
    else:
        ax5.text(0.5, 0.5, '数据文件未找到', transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('')

    # ER网络
    er_second_file = os.path.join(second_row_dir, "第二行δ实验_ER随机网络_数据.txt")
    if os.path.exists(er_second_file):
        er_frequency_data = load_second_row_data_from_txt(er_second_file)
        plot_regulator_target_I_frequency(ax6, er_frequency_data, LINE_CONFIGS[2],
                                          "", False, "ER")
    else:
        ax6.text(0.5, 0.5, '数据文件未找到', transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('')

    # 第三行子图：I节点距离分布
    print("加载第三行子图数据...")

    # BA网络
    ba_third_file = os.path.join(third_row_dir, "第三行ξ实验_BA无标度网络_距离数据.txt")
    if os.path.exists(ba_third_file):
        ba_distance_data = load_third_row_data(ba_third_file)
        plot_I_distance_distribution(ax7, ba_distance_data, LINE_CONFIGS[0],
                                     "", True)
    else:
        ax7.text(0.5, 0.5, '数据文件未找到', transform=ax7.transAxes, ha='center', va='center')
        ax7.set_title('')

    # WS网络
    ws_third_file = os.path.join(third_row_dir, "第三行ξ实验_WS小世界网络_距离数据.txt")
    if os.path.exists(ws_third_file):
        ws_distance_data = load_third_row_data(ws_third_file)
        plot_I_distance_distribution(ax8, ws_distance_data, LINE_CONFIGS[1],
                                     "", False)
    else:
        ax8.text(0.5, 0.5, '数据文件未找到', transform=ax8.transAxes, ha='center', va='center')
        ax8.set_title('')

    # ER网络
    er_third_file = os.path.join(third_row_dir, "第三行ξ实验_ER随机网络_距离数据.txt")
    if os.path.exists(er_third_file):
        er_distance_data = load_third_row_data(er_third_file)
        plot_I_distance_distribution(ax9, er_distance_data, LINE_CONFIGS[2],
                                     "", False)
    else:
        ax9.text(0.5, 0.5, '数据文件未找到', transform=ax9.transAxes, ha='center', va='center')
        ax9.set_title('')

    # 添加子图序号
    subplot_order = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    for ax, label in zip(subplot_order, subplot_labels):
        ax.text(0.5, -0.2, label, transform=ax.transAxes, ha='center', va='top',
                fontsize=14, fontweight='bold')

    # 关键修改：使用极紧凑的布局参数，完全去除留白
    plt.subplots_adjust(
        left=0.12,  # 左边界几乎贴边
        bottom=0.1,  # 底边界稍微留一点空间给子图标签
        right=0.85,  # 右边界几乎贴边
        top=0.9,  # 上边界稍微留一点空间
        wspace=0.14,  # 列间距
        hspace=0.31  # 行间距
    )

    # 强制绘制图形
    plt.draw()
    plt.pause(0.1)

    output_dir = results_dir

    # 关键修改：使用特定的bbox_inches设置来去除留白
    # 方法1：使用tight bounding box但设置极小的padding
    tight_bbox_args = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1,  # 极小的内边距
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'none'
    }

    # 方法2：手动计算bounding box（更精确）
    def get_tight_bbox(fig):
        """获取图形的紧凑边界框"""
        # 获取所有图形的边界框
        bboxes = []
        for ax in fig.axes:
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            bboxes.append(bbox)

        # 计算所有边界框的并集
        from matplotlib.transforms import Bbox
        overall_bbox = Bbox.union(bboxes)
        return overall_bbox

    # 使用方法2：手动bbox
    manual_bbox_args = {
        'bbox_inches': get_tight_bbox(fig),
        'pad_inches': 0,
        'dpi': 300,
        'facecolor': 'white',
        'edgecolor': 'none'
    }

    # 保存PDF - 使用方法1（简单有效）
    output_file_pdf = os.path.join(output_dir, "network_comparison_no_margin.pdf")
    plt.savefig(output_file_pdf, format='pdf', **tight_bbox_args)
    print(f"九幅子图已保存为PDF矢量图（无留白）: {output_file_pdf}")

    # 保存PNG
    output_file_png = os.path.join(output_dir, "network_comparison_no_margin.png")
    plt.savefig(output_file_png, format='png', **tight_bbox_args)
    print(f"九幅子图已保存为高分辨率PNG（无留白）: {output_file_png}")

    # 保存SVG
    output_file_svg = os.path.join(output_dir, "network_comparison_no_margin.svg")
    plt.savefig(output_file_svg, format='svg', **tight_bbox_args)
    print(f"九幅子图已保存为SVG矢量图（无留白）: {output_file_svg}")

    plt.show()


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    # 设置结果目录路径
    results_directory = "results"  # 根据实际情况修改路径
    second_row_directory = "D:\项目py\测试\second_row_results"  # 第二行数据目录
    third_row_directory = "D:\项目py\测试\disan_row_results"

    # 检查目录是否存在
    if not os.path.exists(results_directory):
        print(f"错误: 结果目录 '{results_directory}' 不存在!")
        print("请确保数据文件位于正确目录中")
    else:
        print(f"从目录 '{results_directory}' 加载数据...")
        create_complete_plot_from_data(results_directory, second_row_directory, third_row_directory)
