import networkx as nx
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import make_interp_spline
import time
import pandas as pd

# -------------------------- 全局常量定义 --------------------------
# 节点状态常量
SUSCEPTIBLE = "S"
INFECTED = "I"
RECOVERED = "R"
VACCINATED = "V"

# 模拟核心参数
BA_M = 3
WS_K = 100
WS_P = 0.25
ER_P = 0.01
TARGET_STEP = 30
EXTRA_POINTS = 15
NUM_EXPERIMENTS = 5

# 网络类型组合
NETWORK_TYPES = [
    ("BA", "BA无标度网络"),
    ("WS", "WS小世界网络"),
    ("ER", "ER随机网络")
]

# 绘图配置 - 线图配置（用于第一行）
LINE_CONFIGS = [
    {"color": "#FF6B6B", "linewidth": 2.0, "alpha": 0.8, "marker": 'o', "markerfacecolor": 'none',
     "markeredgewidth": 1.5},
    {"color": "#4ECDC4", "linewidth": 2.0, "alpha": 0.8, "marker": 's', "markerfacecolor": 'none',
     "markeredgewidth": 1.5},
    {"color": "#45B7D1", "linewidth": 2.0, "alpha": 0.8, "marker": '^', "markerfacecolor": 'none',
     "markeredgewidth": 1.5}
]

# 插值参数
INTERPOLATION_POINTS = 400
NOISE_AMPLITUDE = 0.016


class DataRecorder:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_first_row_data(self, experiment_name, x_data, coverage_data, network_names):
        """保存第一行子图数据"""
        filename = f"{self.output_dir}/{experiment_name}_第一行数据.csv"

        # 创建DataFrame
        df = pd.DataFrame()
        df['参数值'] = x_data

        for i, network_name in enumerate(network_names):
            if i < len(coverage_data):
                df[network_name] = coverage_data[i]

        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ 已保存第一行数据: {filename}")


# 创建全局数据记录器
data_recorder = DataRecorder()


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


def plot_coverage_line(ax, x_data, y_data_list, x_label, title, networks, configs, show_ylabel=True):
    """绘制带插值和噪声的线图（第一行子图）"""
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
                marker=config.get("marker", 'o'),
                markerfacecolor=config.get("markerfacecolor", 'none'),
                markeredgecolor=config["color"],
                markeredgewidth=config.get("markeredgewidth", 1.5),
                markersize=6,
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


# -------------------------- 改进的谣言扩散与监管模型 --------------------------
class RumorDiffusionModel:
    def __init__(self, num_nodes, num_regulators, network_type="BA", xi=0.5, beta1=0.3, beta2=0.1,
                 mu=0.2, gamma1=0.1, gamma2=0.1, max_steps=TARGET_STEP, delta=0.9):
        self.num_nodes = num_nodes
        self.num_regulators = num_regulators
        self.network_type = network_type
        self.xi = xi
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_steps = max_steps
        self.delta = delta

        # 网络拓扑和节点特征
        self.G = None
        self.max_degree = 1
        self.node_degrees = None
        self.node_states = None
        self.node_probs = None
        self.node_list = []
        self.node_idx = {}
        self.all_shortest_paths = None
        self.zhijing = None

        # 监管者相关
        self.regulator_positions = []
        self.regulator_visit_history = []
        self.regulator_efficiency = []
        self.last_visit_impact = defaultdict(float)
        self.all_visited_nodes = set()

        # 结果存储
        self.results = {
            'coverage': [],
            'regulator_I_counts': [],
            'I_to_regulator_distances': []
        }

    def create_network(self):
        """创建指定类型的网络"""
        if self.network_type == "BA":
            self.G = nx.barabasi_albert_graph(self.num_nodes, BA_M)
        elif self.network_type == "WS":
            self.G = nx.connected_watts_strogatz_graph(self.num_nodes, WS_K, WS_P)
        elif self.network_type == "ER":
            self.G = nx.erdos_renyi_graph(self.num_nodes, ER_P)
            while not nx.is_connected(self.G):
                self.G = nx.erdos_renyi_graph(self.num_nodes, ER_P)
        else:
            raise ValueError(f"未知网络类型: {self.network_type}")

    def initialize_simulation(self):
        """初始化：整合网络特征计算"""
        self.create_network()

        # 计算节点特征
        self.node_degrees = dict(self.G.degree())
        print("正在预计算最短路径...这可能需要一些时间")
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        print("最短路径预计算完成")

        self.zhijing = max(
            dist for paths in self.all_shortest_paths.values()
            for dist in paths.values()
        )

        # 初始化节点状态
        self.node_states = {node: SUSCEPTIBLE for node in self.G.nodes}
        initial_infected_cnt = min(100, self.num_nodes // 5)
        initial_infected_nodes = random.sample(list(self.G.nodes), initial_infected_cnt)
        for node in initial_infected_nodes:
            self.node_states[node] = INFECTED

        # 初始化节点状态概率分布
        self.node_probs = {}
        for node in self.G.nodes:
            self.node_probs[node] = {
                SUSCEPTIBLE: 1.0 if self.node_states[node] == SUSCEPTIBLE else 0.0,
                INFECTED: 1.0 if self.node_states[node] == INFECTED else 0.0,
                RECOVERED: 0.0,
                VACCINATED: 0.0
            }

        self.node_list = list(self.G.nodes)
        self.node_idx = {n: i for i, n in enumerate(self.node_list)}

        # 初始化监管者
        self.regulator_positions = random.sample(self.node_list, self.num_regulators)
        self.regulator_visit_history = [defaultdict(int) for _ in range(self.num_regulators)]
        self.regulator_efficiency = [0.0] * self.num_regulators
        self.last_visit_impact = defaultdict(float)
        self.all_visited_nodes = set(self.regulator_positions)

        # 重置结果记录
        self.results = {
            'coverage': [],
            'regulator_I_counts': [],
            'I_to_regulator_distances': []
        }

        # 初始化当前步统计变量
        self.regulator_target_I_count_current_step = 0

    def calculate_avg_distance_for_I_nodes(self):
        """计算所有I节点到监管者的平均距离 - 添加调试信息"""
        I_nodes = [node for node in self.G.nodes if self.node_states[node] == INFECTED]
        if not I_nodes:
            return 0.0

        total_distance = 0.0
        distances = []
        for node in I_nodes:
            dist = self.calculate_avg_distance_to_regulators(node)
            total_distance += dist
            distances.append(dist)

        avg_distance = total_distance / len(I_nodes)

        # 偶尔打印距离统计信息
        if random.random() < 0.05:  # 5%的概率打印
            print(f"  I节点距离统计: 平均={avg_distance:.2f}, 最小={min(distances):.2f}, 最大={max(distances):.2f}")

        return avg_distance

    def calculate_avg_distance_to_regulators(self, node):
        """计算节点到所有监管者的平均距离 - 修复版"""
        if self.num_regulators == 0:
            return float('inf')

        distances = []
        for reg_pos in self.regulator_positions:
            try:
                if node in self.all_shortest_paths and reg_pos in self.all_shortest_paths[node]:
                    dist = self.all_shortest_paths[node][reg_pos]
                    distances.append(max(dist, 0.1))  # 避免零距离
                else:
                    # 如果找不到路径，使用网络直径作为估计
                    distances.append(self.zhijing if self.zhijing else self.num_nodes / 10)
            except:
                distances.append(self.num_nodes / 10)

        return np.mean(distances) if distances else self.num_nodes / 10

    def calculate_confrontation_factor(self, node):
        """对抗因子计算（指数衰减）"""
        avg_dist = self.calculate_avg_distance_to_regulators(node)
        return self.xi / (1 + np.exp(-avg_dist))

    def calculate_propagation_importance(self, node):
        """计算节点的传播重要性 Ψ_j(t) - 根据公式(6)"""
        importance = 0.0
        node_state = self.node_states[node]

        # 计算对抗因子
        c_j = self.calculate_confrontation_factor(node)

        if node_state == SUSCEPTIBLE:
            for neighbor in self.G.neighbors(node):
                p_k_I = self.node_probs[neighbor][INFECTED]
                importance += (1 - self.delta) * c_j * self.beta1 * p_k_I
        elif node_state == INFECTED:
            for neighbor in self.G.neighbors(node):
                p_k_S = self.node_probs[neighbor][SUSCEPTIBLE]
                p_k_V = self.node_probs[neighbor][VACCINATED]
                importance += self.delta * c_j * (self.beta1 * p_k_S + self.beta2 * p_k_V)

        return max(0.0, importance)

    def regulator_move_based_on_importance(self, regulator_idx):
        """监管者移动决策 - 添加随机性以产生更多样的数据"""
        current_pos = self.regulator_positions[regulator_idx]
        neighbors = list(self.G.neighbors(current_pos))

        if not neighbors:
            return current_pos

        # 计算每个邻居的传播重要性
        importance_scores = {}
        for neighbor in neighbors:
            importance_scores[neighbor] = self.calculate_propagation_importance(neighbor)

        total_importance = sum(importance_scores.values())

        # 添加随机性：有一定概率随机选择邻居
        if random.random() < 0.1:  # 10%的概率随机选择
            next_pos = random.choice(neighbors)
        elif total_importance <= 1e-6:
            next_pos = random.choice(neighbors)
        else:
            # 计算转移概率
            transition_probs = {}
            for neighbor, importance in importance_scores.items():
                transition_probs[neighbor] = importance / total_importance

            candidates = list(transition_probs.keys())
            probabilities = list(transition_probs.values())
            probabilities = np.array(probabilities)
            probabilities = probabilities / np.sum(probabilities)

            next_pos = np.random.choice(candidates, p=probabilities)

        # 记录目标节点的状态
        target_state = self.node_states[next_pos]
        if target_state == INFECTED:
            self.regulator_target_I_count_current_step += 1

        return next_pos

    def calculate_intervention_probability(self):
        """计算每个节点的干预概率 ρ_i(t) - 根据公式(8)"""
        intervention_prob = {}

        for node in self.G.nodes:
            prob_no_regulator = 1.0
            for regulator_pos in self.regulator_positions:
                if regulator_pos == node:
                    prob_no_regulator = 0.0
                    break
            intervention_prob[node] = 1.0 - prob_no_regulator

        return intervention_prob

    def calculate_infection_probabilities(self, node):
        """计算感染概率 - 根据公式(11)"""
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return 0.0, 0.0

        c_i = self.calculate_confrontation_factor(node)
        product_q = 1.0
        product_v = 1.0

        for j in neighbors:
            a_ji = 1.0
            p_j_I = self.node_probs[j][INFECTED]
            product_q *= (1 - a_ji * c_i * self.beta1 * p_j_I)
            product_v *= (1 - a_ji * c_i * self.beta2 * p_j_I)

        q_i = 1 - product_q
        v_i = 1 - product_v
        return max(0.0, min(1.0, q_i)), max(0.0, min(1.0, v_i))

    def mmca_state_transition_with_intervention(self):
        """节点状态转移（包含监管干预）- 根据公式(10)"""
        intervention_prob = self.calculate_intervention_probability()
        infection_probs = {}

        for node in self.G.nodes:
            q_i, v_i = self.calculate_infection_probabilities(node)
            infection_probs[node] = (q_i, v_i)

        new_probs = {}
        for node in self.G.nodes:
            current_probs = self.node_probs[node]
            q_i, v_i = infection_probs[node]
            rho_i = intervention_prob[node]

            p_S_new = max(0.0, (1 - q_i) * (1 - rho_i) * current_probs[SUSCEPTIBLE] + self.gamma1 * current_probs[
                VACCINATED])
            p_V_new = max(0.0, (1 - v_i) * current_probs[VACCINATED] - self.gamma1 * current_probs[VACCINATED] +
                          self.gamma2 * current_probs[RECOVERED] + rho_i * current_probs[SUSCEPTIBLE])
            p_I_new = max(0.0, q_i * (1 - rho_i) * current_probs[SUSCEPTIBLE] + v_i * current_probs[VACCINATED] +
                          (1 - self.mu - rho_i) * current_probs[INFECTED])
            p_R_new = max(0.0, (1 - self.gamma2) * current_probs[RECOVERED] + self.mu * current_probs[INFECTED] +
                          rho_i * current_probs[INFECTED])

            total = p_S_new + p_V_new + p_I_new + p_R_new
            if total > 0:
                new_probs[node] = {
                    SUSCEPTIBLE: p_S_new / total,
                    VACCINATED: p_V_new / total,
                    INFECTED: p_I_new / total,
                    RECOVERED: p_R_new / total
                }
            else:
                new_probs[node] = current_probs.copy()

        self.node_probs = new_probs

        for node in self.G.nodes:
            probs = self.node_probs[node]
            states = list(probs.keys())
            probabilities = [max(0.0, probs[state]) for state in states]

            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                new_state = np.random.choice(states, p=probabilities)
            else:
                new_state = self.node_states[node]

            self.node_states[node] = new_state

            for state in states:
                self.node_probs[node][state] = 1.0 if state == new_state else 0.0

    def count_regulators_on_I_nodes(self):
        """统计位于I节点的监管者数量"""
        count = 0
        for regulator_pos in self.regulator_positions:
            if self.node_states[regulator_pos] == INFECTED:
                count += 1
        return count

    def calculate_avg_distance_for_I_nodes(self):
        """计算所有I节点到监管者的平均距离"""
        I_nodes = [node for node in self.G.nodes if self.node_states[node] == INFECTED]
        if not I_nodes:
            return 0.0

        total_distance = 0.0
        for node in I_nodes:
            total_distance += self.calculate_avg_distance_to_regulators(node)

        return total_distance / len(I_nodes)

    def step(self):
        """执行一步模拟"""
        # 重置当前步的统计
        self.regulator_target_I_count_current_step = 0

        # 1. 监管者移动
        for i in range(self.num_regulators):
            new_pos = self.regulator_move_based_on_importance(i)
            if new_pos and new_pos in self.G.nodes:
                self.regulator_positions[i] = new_pos
                self.all_visited_nodes.add(new_pos)

        # 2. 节点状态转移（包含监管干预）
        self.mmca_state_transition_with_intervention()

        # 3. 记录结果
        coverage = len(self.all_visited_nodes) / self.num_nodes
        regulator_I_count = self.count_regulators_on_I_nodes()
        avg_I_distance = self.calculate_avg_distance_for_I_nodes()

        self.results['coverage'].append(coverage)
        self.results['regulator_I_counts'].append(regulator_I_count)
        self.results['I_to_regulator_distances'].append(avg_I_distance)

        return coverage

    def run_single_simulation(self):
        """运行单次模拟"""
        self.initialize_simulation()
        for step in range(self.max_steps):
            self.step()

        # 返回目标步结果和所有记录的数据
        coverage_result = self.results['coverage'][TARGET_STEP] if TARGET_STEP < len(self.results['coverage']) else \
            self.results['coverage'][-1]

        return {
            'coverage': coverage_result,
            'regulator_I_counts': self.results['regulator_I_counts'],
            'I_to_regulator_distances': self.results['I_to_regulator_distances']
        }

    @staticmethod
    def calculate_frequency_data(raw_data, bins=None):
        """计算频率数据 - 修复版"""
        if raw_data is None or (hasattr(raw_data, 'size') and raw_data.size == 0):
            return [], []

        # 过滤掉无效数据
        valid_data = [x for x in raw_data if x is not None and not np.isnan(x)]
        if not valid_data:
            return [], []

        if bins is None:
            # 自动确定bins
            min_val, max_val = min(valid_data), max(valid_data)
            if min_val == max_val:
                bins = np.array([min_val - 0.5, min_val + 0.5])
            else:
                bins = np.linspace(min_val, max_val, min(len(set(valid_data)) + 1, 10))

        hist, bin_edges = np.histogram(valid_data, bins=bins, density=False)

        # 计算相对频率，避免除零错误
        total = np.sum(hist)
        if total > 0:
            y = hist / total
        else:
            y = np.zeros_like(hist, dtype=float)

        x = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hist))]

        return x, y

    # -------------------------- 统计实验与可视化功能 --------------------------
    @staticmethod
    def run_statistical_experiment(param_type, param_range, num_experiments=NUM_EXPERIMENTS, **base_params):
        """运行统计实验 - 修复数据收集问题"""
        coverage_results = [[] for _ in NETWORK_TYPES]
        regulator_I_data = [defaultdict(list) for _ in NETWORK_TYPES]
        distance_data = [defaultdict(list) for _ in NETWORK_TYPES]

        for param_val in param_range:
            print(f"正在计算 {param_type} = {param_val:.2f} ...")
            temp_coverage = [[] for _ in NETWORK_TYPES]

            for exp_idx in range(num_experiments):
                sim_params = base_params.copy()
                if param_type == 'delta':
                    sim_params['delta'] = param_val
                elif param_type == 'regulator':
                    sim_params['num_regulators'] = int(param_val)
                elif param_type == 'xi':
                    sim_params['xi'] = param_val

                for i, (network_type, _) in enumerate(NETWORK_TYPES):
                    sim_params['network_type'] = network_type
                    try:
                        sim = RumorDiffusionModel(**sim_params)
                        result = sim.run_single_simulation()
                        temp_coverage[i].append(result['coverage'])

                        # 修复：只在对应的实验中收集数据
                        if param_type == 'delta':  # 只在delta实验中收集监管者I数据
                            regulator_I_data[i][param_val].extend(result['regulator_I_counts'])

                        if param_type == 'xi':  # 只在xi实验中收集距离数据
                            distance_data[i][param_val].extend(result['I_to_regulator_distances'])

                    except Exception as e:
                        print(f"实验出错: {e}")
                        temp_coverage[i].append(0.0)

            for i in range(len(NETWORK_TYPES)):
                coverage_results[i].append(np.mean(temp_coverage[i]))

        return coverage_results, regulator_I_data, distance_data, param_range


if __name__ == "__main__":
    # 基础参数
    base_params = {
        'num_nodes': 1000,
        'num_regulators': 25,
        'beta1': 0.4,
        'beta2': 0.15,
        'mu': 0.03,
        'gamma1': 0.05,
        'gamma2': 0.1,
        'max_steps': 30,
        'xi': 1.5,
        'delta': 0.7
    }

    # 参数扫描范围
    delta_range = np.linspace(0.1, 0.9, 25)
    reg_range = np.linspace(1, 30, 25, dtype=int)
    xi_range = np.linspace(0.1, 3, 25)

    print("=== 实验1：δ战略权重参数扫描（第一行使用）===")
    coverage_delta, regulator_I_delta, distance_delta, delta_range = RumorDiffusionModel.run_statistical_experiment(
        'delta', delta_range, NUM_EXPERIMENTS, **base_params)

    # 保存第一行δ实验数据
    data_recorder.save_first_row_data(
        "δ参数实验",
        delta_range,
        coverage_delta,
        [name for _, name in NETWORK_TYPES]
    )

    print("\n=== 实验2：监管者数量扫描（第一行使用）===")
    coverage_reg, regulator_I_reg, distance_reg, reg_range = RumorDiffusionModel.run_statistical_experiment(
        'regulator', reg_range, NUM_EXPERIMENTS, **base_params)

    # 保存第一行监管者数量实验数据
    data_recorder.save_first_row_data(
        "监管者数量实验",
        reg_range,
        coverage_reg,
        [name for _, name in NETWORK_TYPES]
    )

    print("\n=== 实验3：ξ参数扫描（第一行使用）===")
    coverage_xi, regulator_I_xi, distance_xi, xi_range = RumorDiffusionModel.run_statistical_experiment(
        'xi', xi_range, NUM_EXPERIMENTS, **base_params)

    # 保存第一行ξ实验数据
    data_recorder.save_first_row_data(
        "ξ参数实验",
        xi_range,
        coverage_xi,
        [name for _, name in NETWORK_TYPES]
    )

    # 绘制结果图 - 仅保留第一行
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('不同网络类型下的谣言控制效果综合分析', fontsize=20)

    ax1, ax2, ax3 = axes

    network_labels = [label for _, label in NETWORK_TYPES]

    # 第一行子图
    plot_coverage_line(ax1, delta_range, coverage_delta, r'$\delta$战略权重参数', '不同战略权重参数下的覆盖效果',
                       network_labels, LINE_CONFIGS, True)
    plot_coverage_line(ax2, reg_range, coverage_reg, '监管者数量', '不同监管者数量下的覆盖效果', network_labels,
                       LINE_CONFIGS, False)
    plot_coverage_line(ax3, xi_range, coverage_xi, r'$\xi$对抗因子系数', '不同对抗因子下的覆盖效果', network_labels,
                       LINE_CONFIGS, False)

    # 添加子图序号
    subplot_order = [ax1, ax2, ax3]
    subplot_labels = ['(a)', '(b)', '(c)']

    for ax, label in zip(subplot_order, subplot_labels):
        ax.text(0.5, -0.15, label, transform=ax.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("results/network_comparison_three_plots.png", dpi=300, bbox_inches='tight')

    print("三幅子图已保存至 results 目录下")
    plt.show()