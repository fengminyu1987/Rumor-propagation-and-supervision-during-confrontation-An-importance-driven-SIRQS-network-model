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
BA_M = 10
WS_K = 20
WS_P = 0.15
ER_P = 0.02
TARGET_STEP = 30
EXTRA_POINTS = 15
NUM_EXPERIMENTS = 20

# 网络类型组合
NETWORK_TYPES = [
    ("BA", "BA无标度网络"),
    ("WS", "WS小世界网络"),
    ("ER", "ER随机网络")
]

# 第三行子图参数
THIRD_ROW_XIS = [0, 0.2, 0.5, 0.8, 1.5]
DISTANCE_BINS = 6


class DataRecorder:
    def __init__(self, output_dir="disan_row_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_third_row_data(self, experiment_name, distance_data, network_names):
        """保存第三行子图数据 - 包含距离区间信息到txt文件"""
        for i, (network_type, network_name) in enumerate(NETWORK_TYPES):
            if i < len(distance_data):
                filename = f"{self.output_dir}/{experiment_name}_{network_name}_距离数据.txt"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"{network_name} - 第三行子图距离数据\n")
                    f.write("=" * 50 + "\n\n")

                    # 首先收集所有距离数据来确定区间
                    all_distances = []
                    for xi_val, data_list in distance_data[i].items():
                        if data_list:
                            all_distances.extend(data_list)

                    if not all_distances:
                        f.write("无有效数据\n")
                        continue

                    # 计算距离区间
                    min_dist = min(all_distances)
                    max_dist = max(all_distances)
                    distance_bins = np.linspace(min_dist, max_dist, DISTANCE_BINS + 1)

                    # 生成距离区间标签
                    distance_labels = []
                    for bin_idx in range(DISTANCE_BINS):
                        lower = distance_bins[bin_idx]
                        upper = distance_bins[bin_idx + 1]
                        if upper - lower < 0.1:
                            label = f'{lower:.2f}'
                        else:
                            label = f'{lower:.2f}-{upper:.2f}'
                        distance_labels.append(label)

                    f.write(f"距离区间划分: {distance_labels}\n\n")

                    for xi_val, data_list in distance_data[i].items():
                        if data_list:
                            f.write(f"ξ = {xi_val:.2f}:\n")
                            f.write(f"  原始数据点数量: {len(data_list)}\n")
                            f.write(f"  距离范围: {min(data_list):.3f} - {max(data_list):.3f}\n")
                            f.write(f"  平均距离: {np.mean(data_list):.3f}\n")

                            # 计算频率分布
                            hist, _ = np.histogram(data_list, bins=distance_bins, density=False)
                            total = np.sum(hist)

                            if total > 0:
                                frequencies = hist / total
                                f.write("  距离分布频率:\n")
                                for bin_idx in range(DISTANCE_BINS):
                                    f.write(
                                        f"    {distance_labels[bin_idx]}: {frequencies[bin_idx]:.4f} ({hist[bin_idx]}个数据点)\n")
                            else:
                                f.write("  无有效频率数据\n")
                            f.write("\n")

                    # 添加统计摘要
                    f.write("统计摘要:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"总数据点: {len(all_distances)}\n")
                    f.write(f"全局距离范围: {min_dist:.3f} - {max_dist:.3f}\n")
                    f.write(f"全局平均距离: {np.mean(all_distances):.3f}\n")
                    f.write(f"距离标准差: {np.std(all_distances):.3f}\n")

                print(f"✓ 已保存第三行{network_name}数据: {filename}")


# 创建全局数据记录器
data_recorder = DataRecorder()


# -------------------------- 谣言扩散与监管模型 --------------------------
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
        initial_infected_cnt = max(100, self.num_nodes // 2)
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
        return min(1,self.xi / (1 + np.exp(-avg_dist)))

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
        """监管者移动决策"""
        current_pos = self.regulator_positions[regulator_idx]
        neighbors = list(self.G.neighbors(current_pos))

        if not neighbors:
            return current_pos

        # 计算每个邻居的传播重要性
        importance_scores = {}
        for neighbor in neighbors:
            importance_scores[neighbor] = self.calculate_propagation_importance(neighbor)

        total_importance = sum(importance_scores.values())

        if total_importance <= 1e-6:
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
    def run_third_row_experiment(param_range, num_experiments=NUM_EXPERIMENTS, **base_params):
        """专门运行第三行子图的ξ参数实验"""
        coverage_results = [[] for _ in NETWORK_TYPES]
        distance_data = [defaultdict(list) for _ in NETWORK_TYPES]

        for param_val in param_range:
            print(f"正在计算 ξ = {param_val:.2f} ...")
            temp_coverage = [[] for _ in NETWORK_TYPES]

            for exp_idx in range(num_experiments):
                sim_params = base_params.copy()
                sim_params['xi'] = param_val

                for i, (network_type, _) in enumerate(NETWORK_TYPES):
                    sim_params['network_type'] = network_type
                    try:
                        sim = RumorDiffusionModel(**sim_params)
                        result = sim.run_single_simulation()
                        temp_coverage[i].append(result['coverage'])

                        # 收集距离数据（第三行子图需要）
                        if result['I_to_regulator_distances']:
                            distance_data[i][param_val].extend(result['I_to_regulator_distances'])

                    except Exception as e:
                        print(f"实验出错: {e}")
                        temp_coverage[i].append(0.0)

            for i in range(len(NETWORK_TYPES)):
                coverage_results[i].append(np.mean(temp_coverage[i]))

        return coverage_results, distance_data, param_range


if __name__ == "__main__":
    # 基础参数
    base_params = {
        'num_nodes': 1000,
        'num_regulators': 30,
        'beta1': 0.7,
        'beta2': 0.3,
        'mu': 0.1,
        'gamma1': 0.08,
        'gamma2': 0.15,
        'max_steps': 35,
        'xi': 1.2,
        'delta': 0.6
    }

    # ξ参数扫描范围
    xi_range = np.linspace(0.1, 3, 25)

    print("=== 专门运行第三行子图对应的ξ参数实验 ===")
    coverage_xi, distance_xi, xi_range = RumorDiffusionModel.run_third_row_experiment(
        xi_range, NUM_EXPERIMENTS, **base_params
    )

    # 保存第三行子图数据到txt文件
    data_recorder.save_third_row_data(
        "第三行ξ实验",
        distance_xi,
        [name for _, name in NETWORK_TYPES]
    )

    print("\n=== 实验完成 ===")
    print("所有第三行子图数据已保存到 'disan_row_results' 目录下的txt文件中")