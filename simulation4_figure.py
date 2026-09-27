import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


class ExperimentalDataVisualizer:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.core_data = []
        self.marginal_data = []
        self.optimal_configs = {}
        self.sensitivity_data = {}
        # 定义子图标签 - 现在有6个标签
        self.SUBPLOT_LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    def load_data(self):
        """从txt文件加载数据 - 已适配新数据格式"""
        print("正在加载实验数据...")

        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析核心实验数据
        core_section = content.split("一、核心实验数据")[1].split("二、边际效益分析数据")[0]
        lines = core_section.strip().split('\n')

        current_record = {}
        for line in lines:
            line = line.strip()
            if line.startswith('场景:'):
                if current_record:
                    self.core_data.append(current_record)
                current_record = {'scenario_id': line.split(': ')[1]}
            elif line.startswith('网络规模:'):
                current_record['network_size'] = int(line.split(': ')[1])
            elif line.startswith('传播率:'):
                current_record['infection_rate'] = float(line.split(': ')[1])
            elif line.startswith('监管者数量:'):
                current_record['regulators'] = int(line.split(': ')[1])
            elif line.startswith('最终感染密度:'):
                current_record['final_infection'] = float(line.split(': ')[1])
            elif line.startswith('峰值感染:'):
                current_record['peak_infection'] = float(line.split(': ')[1])
            elif line.startswith('控制效率:'):
                current_record['control_efficiency'] = float(line.split(': ')[1])
            elif line.startswith('成本效益:'):
                current_record['cost_effectiveness'] = float(line.split(': ')[1])
            elif line.startswith('最终感染标准差:'):
                current_record['final_std'] = float(line.split(': ')[1])
            elif line.startswith('峰值感染标准差:'):
                current_record['peak_std'] = float(line.split(': ')[1])

        if current_record:
            self.core_data.append(current_record)

        # 解析边际效益数据
        marginal_section = content.split("二、边际效益分析数据")[1].split("三、最优配置数据")[0]
        lines = marginal_section.strip().split('\n')

        current_record = {}
        for line in lines:
            line = line.strip()
            if line.startswith('网络规模:'):
                if current_record:
                    self.marginal_data.append(current_record)
                parts = line.split(',')
                current_record = {
                    'network_size': int(parts[0].split(': ')[1]),
                    'infection_rate': float(parts[1].split(': ')[1])
                }
            elif line.startswith('监管者范围:'):
                current_record['regulator_range'] = line.split(': ')[1]
            elif line.startswith('边际控制增益:'):
                current_record['marginal_control_gain'] = float(line.split(': ')[1])
            elif line.startswith('边际感染减少:'):
                current_record['marginal_infection_reduction'] = float(line.split(': ')[1])
            elif line.startswith('当前控制水平:'):
                current_record['current_control_level'] = float(line.split(': ')[1])

        if current_record:
            self.marginal_data.append(current_record)

        # 解析最优配置数据
        optimal_section = content.split("三、最优配置数据")[1].split("四、敏感性分析数据")[0]
        lines = optimal_section.strip().split('\n')

        current_scenario = None
        current_type = None
        for line in lines:
            line = line.strip()
            if line.startswith('场景:'):
                current_scenario = line.split(': ')[1]
                self.optimal_configs[current_scenario] = {}
            elif '最优控制配置:' in line:
                current_type = 'best_control'
                self.optimal_configs[current_scenario][current_type] = {}
            elif '最优成本效益配置:' in line:
                current_type = 'best_cost_effectiveness'
                self.optimal_configs[current_scenario][current_type] = {}
            elif line.startswith('监管者数量:'):
                if current_scenario and current_type:
                    self.optimal_configs[current_scenario][current_type]['regulators'] = int(line.split(': ')[1])
            elif line.startswith('控制效率:'):
                if current_scenario and current_type == 'best_control':
                    self.optimal_configs[current_scenario][current_type]['control_efficiency'] = float(
                        line.split(': ')[1])
            elif line.startswith('成本效益:'):
                if current_scenario and current_type == 'best_cost_effectiveness':
                    self.optimal_configs[current_scenario][current_type]['cost_effectiveness'] = float(
                        line.split(': ')[1])

        # 解析敏感性分析数据
        sensitivity_section = content.split("四、敏感性分析数据")[1].split("五、统计检验结果")[0]
        lines = sensitivity_section.strip().split('\n')

        current_type = None
        for line in lines:
            line = line.strip()
            if '网络规模影响:' in line:
                current_type = 'network_size_impact'
                self.sensitivity_data[current_type] = {}
            elif '传播率影响:' in line:
                current_type = 'infection_rate_impact'
                self.sensitivity_data[current_type] = {}
            elif line.startswith('网络规模'):
                if current_type == 'network_size_impact':
                    parts = line.split(':')
                    size = int(parts[0].split(' ')[-1])
                    value = float(parts[1].strip())
                    self.sensitivity_data[current_type][size] = value
            elif line.startswith('传播率'):
                if current_type == 'infection_rate_impact':
                    parts = line.split(':')
                    beta = float(parts[0].split(' ')[-1])
                    value = float(parts[1].strip())
                    self.sensitivity_data[current_type][beta] = value

        print(f"数据加载完成: 核心数据{len(self.core_data)}条, 边际数据{len(self.marginal_data)}条")

    def format_small_number(self, num):
        """格式化小数值，避免显示为0"""
        if isinstance(num, str):
            return num
        if abs(num) < 0.001 and num != 0:
            return f"{num:.2e}"
        else:
            return f"{num:.4f}"

    def plot_main_effect(self, ax, subplot_label):
        """主要效果图：最终感染密度 vs 监管者数量 - 图内legend"""
        df = pd.DataFrame(self.core_data)

        color_palette = sns.color_palette("husl", 6)
        line_styles = ['-', '--', '-.']

        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta)]

                if not subset.empty:
                    color_idx = i * 3 + j
                    linestyle = line_styles[j]
                    label = f'Net{net_size} $\\beta$={beta}'

                    # 误差线显示变异性
                    ax.errorbar(subset['regulators'], subset['final_infection'],
                                yerr=subset['final_std'], marker='o', linewidth=2.5,
                                label=label, capsize=5, capthick=1.5,
                                color=color_palette[color_idx], linestyle=linestyle,
                                alpha=0.7, markersize=7)

        ax.set_xlabel('Number of Supervisors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Infection Density', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=-0.02)

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.18,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def plot_control_efficiency(self, ax, subplot_label):
        """控制效率图 - 将legend放进图内"""
        df = pd.DataFrame(self.core_data)

        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        line_styles = ['-', '--', '-.']
        markers = ['o', 's', '^', 'D', 'v', '<']
        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta)]

                if not subset.empty:
                    color_idx = i * 3 + j
                    marker = markers[color_idx % len(markers)]
                    linestyle = line_styles[j]
                    label = f'Net{net_size} $\\beta$={beta}'

                    ax.plot(subset['regulators'], subset['control_efficiency'],
                            marker=marker, linewidth=3, label=label,
                            color=color_palette[color_idx], linestyle=linestyle,
                            markersize=7, alpha=0.7)

        ax.set_xlabel('Number of Supervisors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Control Efficiency', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.4)

        # 添加零线
        ax.axhline(y=0, color='pink', linestyle='--', alpha=0.7, linewidth=2)

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.18,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def plot_cost_effectiveness(self, ax, subplot_label):
        """成本效益图 - 图内legend"""
        df = pd.DataFrame(self.core_data)

        color_palette = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A', '#3D5A80']
        markers = ['o', 's', '^', 'D', 'v', '<']

        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta) &
                            (df['regulators'] > 0)]

                if not subset.empty:
                    color_idx = i * 3 + j
                    marker = markers[color_idx % len(markers)]
                    label = f'Net{net_size} $\\beta$={beta}'

                    ax.plot(subset['regulators'], subset['cost_effectiveness'],
                            marker=marker, linewidth=2.5, label=label,
                            color=color_palette[color_idx % len(color_palette)],
                            markersize=7, alpha=0.7)

        ax.set_xlabel('Number of Supervisors', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cost-Effectiveness', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.4)

        # 添加零线
        ax.axhline(y=0, color='pink', linestyle='--', alpha=0.7, linewidth=2)

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.18,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def plot_marginal_benefits_400(self, ax, subplot_label):
        """400个节点的边际效益分析图"""
        if not self.marginal_data:
            ax.text(0.5, 0.5, 'No Marginal Benefit Data', ha='center', va='center', transform=ax.transAxes)
            return

        df = pd.DataFrame(self.marginal_data)

        net_size = 400
        colors = {'0.2': '#E63946', '0.4': '#F4A261', '0.6': '#2A9D8F'}
        markers = {'0.2': 'o', '0.4': 's', '0.6': '^'}
        line_styles = {'0.2': '-', '0.4': '--', '0.6': '-.'}

        for beta in [0.2, 0.4, 0.6]:
            subset = df[(df['network_size'] == net_size) &
                        (df['infection_rate'] == beta)]

            if not subset.empty:
                x_positions = []
                for _, row in subset.iterrows():
                    reg_range = row['regulator_range']
                    start, end = map(int, reg_range.split('-'))
                    x_pos = (start + end) / 2
                    x_positions.append(x_pos)

                subset_temp = subset.copy()
                subset_temp['x_position'] = x_positions
                subset_temp = subset_temp.sort_values('x_position')

                x_positions = subset_temp['x_position'].tolist()
                y_values = subset_temp['marginal_control_gain'].tolist()

                label = f'$\\beta$={beta}'
                color = colors[str(beta)]
                marker = markers[str(beta)]
                linestyle = line_styles[str(beta)]

                ax.plot(x_positions, y_values,
                        marker=marker, linewidth=2.5, label=label,
                        color=color, linestyle=linestyle,
                        markersize=8, alpha=0.8, markerfacecolor='none', markeredgewidth=1.2)

        ax.set_xlabel('Number of Supervisors', fontsize=11, fontweight='bold')

        ax.set_ylabel('Marginal Control Benefit', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加零线
        ax.axhline(y=0, color='pink', linestyle='--', alpha=0.7, linewidth=2)

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.18,
            s=subplot_label,
            fontsize=13,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def plot_marginal_benefits_1000(self, ax, subplot_label):
        """1000个节点的边际效益分析图"""
        if not self.marginal_data:
            ax.text(0.5, 0.5, 'No Marginal Benefit Data', ha='center', va='center', transform=ax.transAxes)
            return

        df = pd.DataFrame(self.marginal_data)

        net_size = 1000
        colors = {'0.2': '#8FC3E0', '0.4': '#FFBE9E', '0.6': '#7C9D97'}
        markers = {'0.2': 'o', '0.4': 's', '0.6': '^'}
        line_styles = {'0.2': '-', '0.4': '--', '0.6': '-.'}

        for beta in [0.2, 0.4, 0.6]:
            subset = df[(df['network_size'] == net_size) &
                        (df['infection_rate'] == beta)]

            if not subset.empty:
                x_positions = []
                for _, row in subset.iterrows():
                    reg_range = row['regulator_range']
                    start, end = map(int, reg_range.split('-'))
                    x_pos = (start + end) / 2
                    x_positions.append(x_pos)

                subset_temp = subset.copy()
                subset_temp['x_position'] = x_positions
                subset_temp = subset_temp.sort_values('x_position')

                x_positions = subset_temp['x_position'].tolist()
                y_values = subset_temp['marginal_control_gain'].tolist()

                label = f'$\\beta$={beta}'
                color = colors[str(beta)]
                marker = markers[str(beta)]
                linestyle = line_styles[str(beta)]

                ax.plot(x_positions, y_values,
                        marker=marker, linewidth=2.5, label=label,
                        color=color, linestyle=linestyle,
                        markersize=8, alpha=1, markerfacecolor='none', markeredgewidth=1.2)

        ax.set_xlabel('Number of Supervisors', fontsize=11, fontweight='bold')

        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加零线
        ax.axhline(y=0, color='pink', linestyle='--', alpha=0.7, linewidth=2)

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.18,
            s=subplot_label,
            fontsize=13,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def plot_sensitivity_analysis(self, ax, subplot_label):
        """敏感性分析图 - 图内legend并增加y轴最大值"""
        size_data = self.sensitivity_data.get('网络规模影响', {})
        beta_data = self.sensitivity_data.get('传播率影响', {})

        if not size_data and not beta_data:
            size_data = self.sensitivity_data.get('network_size_impact', {})
            beta_data = self.sensitivity_data.get('infection_rate_impact', {})

        if not size_data and not beta_data:
            ax.text(0.5, 0.5, 'No Sensitivity Data', ha='center', va='center', transform=ax.transAxes)
            return

        categories = []
        values = []
        colors = []

        if size_data:
            for size, value in size_data.items():
                categories.append(f'N={size}')
                values.append(max(value, 1e-10))
                colors.append('skyblue')

        if beta_data:
            for beta, value in beta_data.items():
                categories.append(f'$\\beta$={beta}')
                values.append(max(value, 1e-10))
                colors.append('lightcoral')

        x = np.arange(len(categories))

        bars = ax.bar(x, values, alpha=0.7, color=colors, edgecolor='black')

        if values:
            y_max = max(values) * 1.2
            ax.set_ylim(0, y_max)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    self.format_small_number(value), ha='center', va='bottom', fontsize=9)

        legend_elements = [
            Patch(facecolor='skyblue', alpha=0.7, label='Network Size Impact'),
            Patch(facecolor='lightcoral', alpha=0.7, label='$\\beta$ Impact')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='best')

        ax.set_xlabel('Simulation Conditions', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Control Efficiency', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.grid(True, alpha=0.3, axis='y')

        # 在子图下方添加标签
        ax.text(
            x=0.5, y=-0.22,
            s=subplot_label,
            fontsize=14,
            ha='center', va='top',
            transform=ax.transAxes
        )

    def create_comprehensive_visualization(self, hspace=0.32, wspace=0.4,
                                           top_margin=0.92, bottom_margin=0.15,
                                           left_margin=0.02, right_margin=0.98,
                                           output_format='pdf'):
        """
        创建综合可视化图表 - 新布局：第一行两个子图填满宽度，第二行三个子图，第三行一个子图

        参数:
        hspace: 行间距 (默认0.3)
        wspace: 列间距 (默认0.25)
        top_margin: 顶部边距 (默认0.92)
        bottom_margin: 底部边距 (默认0.15)
        left_margin: 左边距 (默认0.02)
        right_margin: 右边距 (默认0.98)
        output_format: 输出格式，支持 'pdf', 'svg', 'eps' 等矢量图格式
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")

        # 设置字体为可嵌入矢量图的字体
        plt.rcParams['pdf.fonttype'] = 42  # 确保字体在PDF中可编辑
        plt.rcParams['ps.fonttype'] = 42  # 确保字体在PostScript中可编辑
        plt.rcParams['svg.fonttype'] = 'path'  # 确保字体在SVG中作为路径嵌入

        # 新布局：3行，使用更灵活的宽度比例
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(
            nrows=3, ncols=6,  # 使用6列，方便分配
            figure=fig,
            height_ratios=[1, 1, 0.8],  # 第三行稍矮
            hspace=hspace,  # 使用可调参数
            wspace=wspace  # 使用可调参数
        )

        # 第一行：两个图各占3列（6/2=3）
        ax_a = fig.add_subplot(gs[0, 0:3])  # 前3列
        ax_b = fig.add_subplot(gs[0, 3:6])  # 后3列

        # 第二行：三个图各占2列（6/3=2）
        ax_c = fig.add_subplot(gs[1, 0:2])  # 前2列
        ax_d1 = fig.add_subplot(gs[1, 2:4])  # 中间2列
        ax_d2 = fig.add_subplot(gs[1, 4:6])  # 后2列

        # 第三行：一个图占全部6列
        ax_e = fig.add_subplot(gs[2, :])

        axes = [ax_a, ax_b, ax_c, ax_d1, ax_d2, ax_e]
        # 绘制各个子图
        self.plot_main_effect(axes[0], self.SUBPLOT_LABELS[0])  # (a)
        self.plot_control_efficiency(axes[1], self.SUBPLOT_LABELS[1])  # (b)
        self.plot_cost_effectiveness(axes[2], self.SUBPLOT_LABELS[2])  # (c)
        self.plot_marginal_benefits_400(axes[3], self.SUBPLOT_LABELS[3])  # (d1)
        self.plot_marginal_benefits_1000(axes[4], self.SUBPLOT_LABELS[4])  # (d2)
        self.plot_sensitivity_analysis(axes[5], self.SUBPLOT_LABELS[5])  # (e)

        # 调整布局，使用可调参数
        plt.tight_layout(rect=[left_margin, bottom_margin, right_margin, top_margin])

        # 保存矢量图
        save_path = f'监管有效性深度分析_基于传播潜力_完整六图.{output_format}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                    format=output_format)
        print(f"矢量图已保存至：{save_path}")

        plt.show()


# 使用示例
if __name__ == "__main__":
    # 请替换为您的数据文件路径
    data_file = r"D:\项目py\测试\simulation_data\all_experimental_data.txt"

    # 创建可视化器
    visualizer = ExperimentalDataVisualizer(data_file)

    # 加载数据
    visualizer.load_data()

    # 生成综合图表 - 输出PDF格式矢量图
    visualizer.create_comprehensive_visualization(output_format='pdf')

    # 也可以选择输出SVG格式
    # visualizer.create_comprehensive_visualization(output_format='svg')

    # 或者EPS格式（适合LaTeX文档）
    # visualizer.create_comprehensive_visualization(output_format='eps')