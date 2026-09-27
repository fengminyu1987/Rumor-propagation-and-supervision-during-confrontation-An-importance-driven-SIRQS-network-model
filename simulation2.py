import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
from datetime import datetime
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 全局常量定义 --------------------------
# 节点状态常量
SUSCEPTIBLE = "S"
INFECTED = "I"
RECOVERED = "R"
VACCINATED = "V"

# 模拟核心参数
BA_M = 6
NUM_NODES = 500
NUM_REGULATORS = 25
MAX_STEPS = 200
INITIAL_INFECTED_RATIO = 0.12
NUM_REPEATS = 3  # 热力图重复次数较少以加快速度

# 自定义色图（用于热力图）
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

# 全局绘图样式配置
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
    'lines.linewidth': 1.2
})


class SIVRSimulation:
    def __init__(self, num_nodes=NUM_NODES, avg_degree=BA_M, num_regulators=NUM_REGULATORS,
                 beta1=0.5, beta2=0.2,
                 mu=0.2,  # I→R的转移系数（公式中μ）
                 gamma1=0.1,  # V→S的转移系数（公式中γ₁）
                 gamma2=0.08,  # R→V的转移系数（公式中γ₂）
                 xi=1,  # 对抗效应系数（公式中ξ）
                 max_steps=MAX_STEPS,
                 w_topology=0.2, w_risk=0.3, w_spread=0.3,
                 mmax=15, delta=0.8):  # 战略权重参数δ
        """SIVR模型初始化"""
        # 状态映射
        self.STATE = {
            'S': 0,  # 易感状态
            'V': 1,  # 免疫状态
            'I': 2,  # 感染状态
            'R': 3  # 恢复状态
        }

        # 基础网络参数
        self.num_nodes = num_nodes
        self.ba_m = avg_degree
        self.num_regulators = num_regulators
        self.max_steps = max_steps

        # 模型核心参数
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.xi = xi
        self.delta = delta
        self.zhijing = None

        # 监管策略参数
        self.w_topology = w_topology
        self.w_risk = w_risk
        self.w_spread = w_spread
        self.mmax = mmax

        # 网络与节点特征
        self.all_shortest_paths = None
        self.states = None
        self.node_probs = None

        # 监管者相关
        self.regulator_positions = []
        self.regulator_visit_history = []
        self.regulator_efficiency = []
        self.last_visit_impact = defaultdict(float)
        self.last_visit_time = defaultdict(int)

        # 结果记录
        self.results = {
            'S': [], 'I': [], 'V': [], 'R': [],
            'regulator_efficiency': []
        }

    def initialize_simulation(self):
        """初始化模拟环境"""
        # 创建连通BA网络
        self.G = nx.barabasi_albert_graph(self.num_nodes, self.ba_m)
        while not nx.is_connected(self.G):
            self.G = nx.barabasi_albert_graph(self.num_nodes, self.ba_m)

        # 预计算所有节点间的最短路径长度
        print("正在预计算最短路径...")
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        print("最短路径预计算完成")
        self.zhijing = max(
            dist for paths in self.all_shortest_paths.values()
            for dist in paths.values()
        )

        # 初始化节点状态
        self.states = np.full(self.num_nodes, self.STATE['S'], dtype=int)
        initial_infected_cnt = int(self.num_nodes * INITIAL_INFECTED_RATIO)
        initial_infected_nodes = random.sample(list(self.G.nodes), initial_infected_cnt)
        for node in initial_infected_nodes:
            self.states[node] = self.STATE['I']

        # 初始化节点状态概率分布
        self.node_probs = {}
        for node in self.G.nodes:
            self.node_probs[node] = np.zeros(4)
            if self.states[node] == self.STATE['S']:
                self.node_probs[node][self.STATE['S']] = 1.0
            elif self.states[node] == self.STATE['I']:
                self.node_probs[node][self.STATE['I']] = 1.0
            elif self.states[node] == self.STATE['R']:
                self.node_probs[node][self.STATE['R']] = 1.0
            elif self.states[node] == self.STATE['V']:
                self.node_probs[node][self.STATE['V']] = 1.0

        # 初始化监管者
        if self.num_regulators > 0:
            self.regulator_positions = random.sample(list(self.G.nodes), self.num_regulators)
            self.regulator_visit_history = [defaultdict(int) for _ in range(self.num_regulators)]
            self.regulator_efficiency = [0.0] * self.num_regulators
        else:
            self.regulator_positions = []
            self.regulator_visit_history = []
            self.regulator_efficiency = [0.0]

        self.last_visit_impact.clear()
        self.last_visit_time.clear()
        for node in self.G.nodes:
            self.last_visit_time[node] = -1000

        # 记录初始状态
        self.record_state_density()

    def record_state_density(self):
        """记录当前各状态密度"""
        total = self.num_nodes
        s_density = np.sum(self.states == self.STATE['S']) / total
        i_density = np.sum(self.states == self.STATE['I']) / total
        v_density = np.sum(self.states == self.STATE['V']) / total
        r_density = np.sum(self.states == self.STATE['R']) / total

        self.results['S'].append(s_density)
        self.results['I'].append(i_density)
        self.results['V'].append(v_density)
        self.results['R'].append(r_density)

        if len(self.results['regulator_efficiency']) == 0:
            self.results['regulator_efficiency'].append(0.0)

    def calculate_avg_distance_to_regulators(self, node):
        """计算节点到所有监管者的平均距离（使用预计算结果）"""
        if self.num_regulators == 0:
            return 0

        distances = []
        for reg_pos in self.regulator_positions:
            if reg_pos in self.all_shortest_paths and node in self.all_shortest_paths[reg_pos]:
                dist = self.all_shortest_paths[reg_pos][node]
                distances.append(max(dist, 1e-6))
            else:
                distances.append(self.num_nodes)
        return np.mean(distances) if distances else self.num_nodes

    def calculate_confrontation_factor(self, node):
        """计算对抗因子（指数衰减）"""
        avg_dist = self.calculate_avg_distance_to_regulators(node)
        return min(1, self.xi / (1 + np.exp(-avg_dist)))

    def calculate_propagation_importance(self, node_j, current_step):
        """计算节点j的传播重要性Ψ_j(t) - 公式(6)"""
        if self.states[node_j] == self.STATE['S']:
            # S状态节点：预防重要性
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_I = self.node_probs[neighbor_k][self.STATE['I']]
                importance += c_j * self.beta1 * p_k_I * 1.0
            return (1 - self.delta) * importance

        elif self.states[node_j] == self.STATE['I']:
            # I状态节点：干预重要性
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_S = self.node_probs[neighbor_k][self.STATE['S']]
                p_k_V = self.node_probs[neighbor_k][self.STATE['V']]
                term = c_j * self.beta1 * p_k_S + c_j * self.beta2 * p_k_V
                importance += term * 1.0
            return self.delta * importance

        else:
            return 0.0

    def regulator_next_move(self, regulator_idx):
        """监管者移动决策 - 基于传播重要性的随机游走"""
        current_pos = self.regulator_positions[regulator_idx]
        neighbors = list(self.G.neighbors(current_pos))
        if not neighbors:
            return current_pos

        # 计算每个邻居的传播重要性
        psi_values = []
        for neighbor in neighbors:
            psi = self.calculate_propagation_importance(neighbor, len(self.results['S']) - 1)
            psi_values.append(psi)

        # 计算转移概率
        total_psi = sum(psi_values)
        if total_psi == 0:
            transition_probs = [1.0 / len(neighbors)] * len(neighbors)
        else:
            transition_probs = [psi / total_psi for psi in psi_values]

        # 根据概率选择下一个位置
        next_pos = np.random.choice(neighbors, p=transition_probs)
        return next_pos

    def calculate_intervention_probability(self, node_i):
        """计算节点i的干预概率ρ_i(t)"""
        if self.num_regulators == 0:
            return 0.0

        prob_no_regulator = 1.0
        for k in range(self.num_regulators):
            regulator_pos = self.regulator_positions[k]
            if regulator_pos == node_i:
                prob_no_regulator *= 0.0
                break

        return 1.0 - prob_no_regulator

    def execute_regulatory_action(self, node):
        """执行监管动作"""
        original_state = self.states[node]

        if original_state == self.STATE['S']:
            # S → V
            self.states[node] = self.STATE['V']
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE['V']] = 1.0
            return 0.7
        elif original_state == self.STATE['I']:
            # I → R
            self.states[node] = self.STATE['R']
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE['R']] = 1.0
            return 1.0
        else:
            return 0.3

    def calculate_infection_probabilities(self, node):
        """计算感染概率（q_i和v_i）"""
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return 0.0, 0.0

        c_i = self.calculate_confrontation_factor(node)
        product_q = 1.0
        product_v = 1.0

        for neighbor in neighbors:
            a_ji = 1.0
            p_j_I = self.node_probs[neighbor][self.STATE['I']]
            product_q *= (1 - a_ji * c_i * self.beta1 * p_j_I)
            product_v *= (1 - a_ji * c_i * self.beta2 * p_j_I)

        q_i = 1 - product_q
        v_i = 1 - product_v
        return max(0.0, min(1.0, q_i)), max(0.0, min(1.0, v_i))

    def update_regulators(self):
        """更新监管者位置并执行监管动作"""
        if self.num_regulators == 0:
            self.results['regulator_efficiency'].append(0.0)
            return

        total_efficiency = 0
        current_step = len(self.results['S']) - 1

        # 更新监管者位置
        new_positions = []
        for i in range(self.num_regulators):
            new_pos = self.regulator_next_move(i)
            new_positions.append(new_pos)

            # 记录访问历史
            self.regulator_visit_history[i][new_pos] = current_step
            self.last_visit_time[new_pos] = current_step

            # 计算移动效率
            psi_target = self.calculate_propagation_importance(new_pos, current_step)
            self.regulator_efficiency[i] = psi_target
            total_efficiency += psi_target

        self.regulator_positions = new_positions

        # 执行监管干预
        for i, pos in enumerate(self.regulator_positions):
            self.execute_regulatory_action(pos)

        # 记录平均监管效率
        avg_efficiency = total_efficiency / self.num_regulators
        self.results['regulator_efficiency'].append(avg_efficiency)

    def update_mmca_states(self):
        """更新所有节点的状态（整合监管干预的MMCA机制）"""
        # 计算每个节点的干预概率
        intervention_probs = {}
        for node in self.G.nodes:
            intervention_probs[node] = self.calculate_intervention_probability(node)

        for node in self.G.nodes:
            rho_i = intervention_probs[node]
            current_state = self.states[node]

            # 计算感染概率
            q_i, v_i = self.calculate_infection_probabilities(node)

            # 根据当前状态和干预概率更新状态概率
            if current_state == self.STATE['S']:
                # S → I (感染) 或 S → V (监管干预)
                if np.random.random() < rho_i:
                    # 被监管干预：S → V
                    self.states[node] = self.STATE['V']
                    self.node_probs[node][self.STATE['V']] = 1.0
                else:
                    # 正常感染过程
                    if np.random.random() < q_i:
                        self.states[node] = self.STATE['I']
                        self.node_probs[node][self.STATE['I']] = 1.0
                    else:
                        self.states[node] = self.STATE['S']
                        self.node_probs[node][self.STATE['S']] = 1.0

            elif current_state == self.STATE['V']:
                # V → S (失效) 或 V → I (突破感染)
                if np.random.random() < self.gamma1:
                    self.states[node] = self.STATE['S']
                    self.node_probs[node][self.STATE['S']] = 1.0
                elif np.random.random() < v_i:
                    self.states[node] = self.STATE['I']
                    self.node_probs[node][self.STATE['I']] = 1.0
                else:
                    self.states[node] = self.STATE['V']
                    self.node_probs[node][self.STATE['V']] = 1.0

            elif current_state == self.STATE['I']:
                # I → R (自然恢复或监管干预)
                if np.random.random() < rho_i:
                    # 被监管干预：I → R
                    self.states[node] = self.STATE['R']
                    self.node_probs[node][self.STATE['R']] = 1.0
                elif np.random.random() < self.mu:
                    # 自然恢复：I → R
                    self.states[node] = self.STATE['R']
                    self.node_probs[node][self.STATE['R']] = 1.0
                else:
                    self.states[node] = self.STATE['I']
                    self.node_probs[node][self.STATE['I']] = 1.0

            elif current_state == self.STATE['R']:
                # R → V (转化为免疫)
                if np.random.random() < self.gamma2:
                    self.states[node] = self.STATE['V']
                    self.node_probs[node][self.STATE['V']] = 1.0
                else:
                    self.states[node] = self.STATE['R']
                    self.node_probs[node][self.STATE['R']] = 1.0

    def step(self):
        """执行一步模拟"""
        # 先更新监管者位置和干预
        self.update_regulators()

        # 然后更新节点状态（包含监管干预的影响）
        self.update_mmca_states()

        # 记录状态密度
        self.record_state_density()

    def run_simulation(self):
        """运行完整模拟"""
        self.initialize_simulation()
        for step in range(self.max_steps - 1):
            if step % 50 == 0:  # 每50步显示一次进度
                print(f"  模拟进度: {step + 1}/{self.max_steps}")
            self.step()
        return self.results

    def run_until_target(self, target_rate=0.1):
        """运行模拟直到感染率低于目标值，返回所需步数"""
        self.initialize_simulation()
        for step in range(self.max_steps):
            self.step()
            current_infect_rate = np.sum(self.states == self.STATE['I']) / self.num_nodes
            if current_infect_rate <= target_rate:
                return step
        return self.max_steps  # 未达到目标返回最大步数


def run_heatmap_experiments():
    """运行热力图实验并绘制六个热力图"""
    # 基础参数
    base_params = {
        'num_nodes': NUM_NODES,
        'avg_degree': BA_M,
        'num_regulators': NUM_REGULATORS,
        'max_steps': MAX_STEPS,
        'w_topology': 0.2,
        'w_risk': 0.3,
        'w_spread': 0.3,
        'mmax': 15
    }

    # 六个参数的扫描配置 - 修复字符串转义问题
    param_configs = [
        {
            'name': 'mu',
            'range': np.linspace(0.2, 0.6, 15),  # I→R康复率
            'label': 'μ (I→R康复率)',
            'latex': r'$\mu$'  # 使用原始字符串
        },
        {
            'name': 'gamma1',
            'range': np.linspace(0, 0.1, 15),  # V→S免疫消退率
            'label': 'γ₁ (V→S免疫消退率)',
            'latex': r'$\gamma_1$'  # 使用原始字符串
        },
        {
            'name': 'gamma2',
            'range': np.linspace(0.01, 0.06, 15),  # R→V免疫消退率
            'label': 'γ₂ (R→V免疫消退率)',
            'latex': r'$\gamma_2$'  # 使用原始字符串
        },
        {
            'name': 'xi',
            'range': np.linspace(0.1, 1, 15),  # 谣言对抗系数
            'label': 'ξ (谣言对抗系数)',
            'latex': r'$\xi$'  # 使用原始字符串
        },
        {
            'name': 'delta',
            'range': np.linspace(0.1, 0.9, 15),  # 战略权重参数
            'label': 'δ (战略权重参数)',
            'latex': r'$\delta$'  # 使用原始字符串
        },
        {
            'name': 'theta',
            'range': np.linspace(0, 0.5, 15),  # β2/β1比值
            'label': 'θ (β₂/β₁比值)',
            'latex': r'$\theta$'  # 使用原始字符串
        }
    ]

    # 第一行子图的β1范围（保持不变）
    beta1_range_row1 = np.linspace(0.1, 0.8, 12)
    # 第二行子图的β1范围（改为0到0.3）
    beta1_range_row2 = np.linspace(0.0, 0.3, 12)
    target_rate = 0.02  # 目标感染率

    # 存储所有结果
    all_results = []

    print("开始热力图参数扫描实验...")

    for config_idx, config in enumerate(param_configs):
        print(f"\n扫描参数 {config_idx + 1}/6: {config['label']}")

        # 根据子图位置选择β1范围
        if config_idx < 3:  # 第一行子图（前3个参数）
            beta1_range = beta1_range_row1
        else:  # 第二行子图（后3个参数）
            beta1_range = beta1_range_row2

        # 初始化结果矩阵
        steps_matrix = np.full((len(beta1_range), len(config['range'])), MAX_STEPS, dtype=np.float64)

        for beta1_idx, beta1 in enumerate(beta1_range):
            for param_idx, param_val in enumerate(config['range']):
                # 设置基础参数
                current_params = base_params.copy()
                current_params['beta1'] = beta1

                # 特殊处理theta参数
                if config['name'] == 'theta':
                    # theta = beta2 / beta1, 所以 beta2 = theta * beta1
                    beta2 = param_val * beta1
                    current_params['beta2'] = beta2
                    # 根据参数索引设置不同的默认参数
                    if config_idx == 1:  # 图b (gamma1)
                        current_params['mu'] = 0.3
                        current_params['gamma1'] = 0.05  # 这个会被扫描覆盖，但设置默认值
                        current_params['gamma2'] = 0.06
                        current_params['xi'] = 0.25
                        current_params['delta'] = 0.8
                        current_params['beta2'] = 0.05  # 覆盖上面的beta2
                    elif config_idx == 4:  # 图e (delta)
                        current_params['mu'] = 0.3
                        current_params['gamma1'] = 0.06
                        current_params['gamma2'] = 0.05
                        current_params['xi'] = 0.3
                        current_params['delta'] = 0.8  # 这个会被扫描覆盖，但设置默认值
                        current_params['beta2'] = 0.1  # 覆盖上面的beta2
                    else:
                        # 其他图的默认参数
                        current_params['mu'] = 0.2
                        current_params['gamma1'] = 0.05
                        current_params['gamma2'] = 0.03
                        current_params['xi'] = 1.5
                        current_params['delta'] = 0.7
                else:
                    # 设置当前扫描的参数
                    current_params[config['name']] = param_val

                    # 根据参数索引设置不同的默认参数
                    if config_idx == 1:  # 图b (gamma1)
                        current_params['mu'] = 0.3
                        current_params['gamma2'] = 0.03
                        current_params['xi'] = 0.25
                        current_params['delta'] = 0.8
                        current_params['beta2'] = 0.15
                        # gamma1 由扫描参数设置，不在这里设置
                    elif config_idx == 4:  # 图e (delta)
                        current_params['mu'] = 0.3
                        current_params['gamma1'] = 0.06
                        current_params['gamma2'] = 0.05
                        current_params['xi'] = 0.3
                        current_params['beta2'] = 0.1
                        # delta 由扫描参数设置，不在这里设置
                    else:
                        # 其他图的默认参数
                        if config['name'] != 'mu':
                            current_params['mu'] = 0.2
                        if config['name'] != 'gamma1':
                            current_params['gamma1'] = 0.05
                        if config['name'] != 'gamma2':
                            current_params['gamma2'] = 0.03
                        if config['name'] != 'xi':
                            current_params['xi'] = 1.5
                        if config['name'] != 'delta':
                            current_params['delta'] = 0.7
                        current_params['beta2'] = 0.1  # 默认beta2值

                # 多次重复实验取平均
                total_steps = 0
                for repeat in range(NUM_REPEATS):
                    simulator = SIVRSimulation(**current_params)
                    steps_needed = simulator.run_until_target(target_rate)
                    total_steps += steps_needed

                avg_steps = total_steps / NUM_REPEATS
                steps_matrix[beta1_idx, param_idx] = avg_steps

                # 打印进度
                if (param_idx + 1) % 5 == 0 or param_idx == len(config['range']) - 1:
                    print(f"  β1={beta1:.2f} | {config['label']}={param_val:.3f} | 平均步数: {avg_steps:.1f}")

        all_results.append((steps_matrix, config, beta1_range))

    # 绘制热力图
    plot_heatmaps(all_results, target_rate)

    # 保存数据
    save_heatmap_data(all_results, target_rate)

    return all_results


def plot_heatmaps(all_results, target_rate):
    """绘制六个热力图的函数"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 找到所有结果中的最小和最大值，用于统一色标范围
    all_min = min(np.min(result[0]) for result in all_results)
    all_max = max(np.max(result[0]) for result in all_results)

    for idx, (steps_matrix, config, beta1_range) in enumerate(all_results):
        ax = axes[idx]

        # 平滑热力图数据
        smoothed_matrix = gaussian_filter(steps_matrix, sigma=1.0)

        # 绘制热力图
        im = ax.imshow(
            smoothed_matrix,
            cmap=custom_cmap,
            aspect='auto',
            vmin=all_min,
            vmax=all_max,
            interpolation='spline16',
            origin='lower',
            extent=[0, 1, 0, 1]
        )

        # 设置坐标轴标签和刻度
        param_range = config['range']
        x_ticks = np.linspace(0, 1, 5)
        x_tick_positions = np.linspace(0, len(param_range) - 1, 5)
        x_tick_labels = [f'{param_range[int(pos)]:.2f}' for pos in x_tick_positions]

        y_ticks = np.linspace(0, 1, 5)
        y_tick_positions = np.linspace(0, len(beta1_range) - 1, 5)
        y_tick_labels = [f'{beta1_range[int(pos)]:.2f}' for pos in y_tick_positions]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        # 设置轴标签
        ax.set_xlabel(f"{config['latex']} ({config['label']})", fontsize=12, labelpad=10)
        ax.set_ylabel('β₁ (S→I传播概率)', fontsize=12, labelpad=10)

        # 设置标题
        ax.set_title(f'{config["latex"]}对控制速度的影响\n(目标感染率≤{target_rate})',
                     fontsize=14, pad=15)

        # 添加等高线


    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'感染率首次≤{target_rate}的步数', rotation=270, labelpad=20, fontsize=12)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # 添加总标题
    fig.suptitle('SIVR模型参数敏感性分析热力图', fontsize=16, y=0.98)

    # 保存图像
    if not os.path.exists("heatmap_results"):
        os.makedirs("heatmap_results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"heatmap_results/sivr_heatmaps_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"heatmap_results/sivr_heatmaps_{timestamp}.pdf", bbox_inches='tight')

    print(f"\n热力图已保存至 heatmap_results 文件夹")
    plt.show()


def save_heatmap_data(all_results, target_rate):
    """保存热力图数据到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"heatmap_results/sivr_heatmap_data11_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("SIVR模型热力图实验数据\n")
        f.write("=" * 50 + "\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目标感染率: {target_rate}\n")
        f.write(f"重复实验次数: {NUM_REPEATS}\n")
        f.write(f"最大步数: {MAX_STEPS}\n")
        f.write("=" * 50 + "\n\n")

        for idx, (steps_matrix, config, beta1_range) in enumerate(all_results):
            f.write(f"参数 {idx + 1}: {config['label']} ({config['latex']})\n")
            f.write(f"参数范围: {config['range'][0]:.3f} - {config['range'][-1]:.3f}\n")
            f.write(f"β₁范围: {beta1_range[0]:.2f} - {beta1_range[-1]:.2f}\n")
            f.write("步数矩阵 (行: β₁值, 列: 参数值):\n")

            # 写入列标签 (参数值)
            f.write("β₁\\Param\t")
            for j, param_val in enumerate(config['range'][::2]):  # 每隔一个值显示
                f.write(f"{param_val:.3f}\t")
            f.write("\n")

            # 写入数据行
            for i, beta1 in enumerate(beta1_range[::2]):  # 每隔一个值显示
                f.write(f"{beta1:.2f}\t\t")
                for j in range(0, len(config['range']), 2):
                    f.write(f"{steps_matrix[i * 2, j]:.1f}\t")
                f.write("\n")
            f.write("\n" + "-" * 50 + "\n\n")

    print(f"实验数据已保存至: {filename}")


if __name__ == "__main__":
    # 运行热力图实验
    results = run_heatmap_experiments()