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
WS_K = 100
WS_P = 0.05
ER_P = 0.02
TARGET_STEP = 30
EXTRA_POINTS = 15
NUM_EXPERIMENTS = 5

# 网络类型组合
NETWORK_TYPES = [
    ("BA", "BA无标度网络"),
    ("WS", "WS小世界网络"),
    ("ER", "ER随机网络")
]

# 第二行子图参数 - 只保留三个关键值
SECOND_ROW_DELTAS = [0.1, 0.3, 0.7]


class DataRecorder:
    def __init__(self, output_dir="second_row_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_second_row_data(self, experiment_name, regulator_target_I_data, network_names):
        """保存第二行子图数据 - 修复版"""
        for i, (network_type, network_name) in enumerate(NETWORK_TYPES):
            if i < len(regulator_target_I_data):
                filename = f"{self.output_dir}/{experiment_name}_{network_name}_数据.txt"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"{network_name} - 监管者目标I节点百分比数据\n")
                    f.write("=" * 50 + "\n\n")

                    for delta_val, data_list in regulator_target_I_data[i].items():
                        if data_list:
                            # 将比例转换为百分比
                            percentage_data = [x * 100 for x in data_list]

                            f.write(f"Delta参数值: {delta_val:.2f}\n")
                            f.write(f"数据点数量: {len(percentage_data)}\n")
                            f.write("具体数据值:\n")

                            # 每行写10个数据
                            for j in range(0, len(percentage_data), 10):
                                batch = percentage_data[j:j + 10]
                                line = " ".join([f"{x:.2f}" for x in batch])
                                f.write(line + "\n")

                            # 统计信息
                            unique_values, counts = np.unique(percentage_data, return_counts=True)
                            total_count = len(percentage_data)
                            frequencies = counts / total_count

                            f.write(f"\n统计信息:\n")
                            f.write(f"唯一值数量: {len(unique_values)}\n")
                            f.write(f"频率分布:\n")
                            for val, freq in zip(unique_values, frequencies):
                                f.write(f"  值 {val:.2f}: 频率 {freq:.4f}\n")

                            f.write("\n" + "-" * 40 + "\n\n")

                print(f"✓ 已保存{network_name}数据: {filename}")


# 创建全局数据记录器
data_recorder = DataRecorder()


class SecondRowExperiment:
    """专门用于第二行子图（监管者向I节点漂移频率分布）的实验配置"""

    @staticmethod
    def get_base_params():
        """第二行子图的专用参数 - 修复版"""
        return {
            'num_nodes': 1000,
            'num_regulators': 100,
            'beta1': 0.5,
            'beta2': 0.2,
            'mu': 0.08,
            'gamma1': 0.15,
            'gamma2': 0.22,
            'max_steps': 100,
            'xi': 1.2,
            'delta': 0.8
        }

    @staticmethod
    def initialize_simulation_with_custom_params(model):
        """使用第二行专用参数初始化模拟 - 修复版"""
        model.create_network()

        # 计算节点特征
        model.node_degrees = dict(model.G.degree())
        print("正在预计算最短路径...这可能需要一些时间")
        model.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(model.G))
        print("最短路径预计算完成")

        model.zhijing = max(
            dist for paths in model.all_shortest_paths.values()
            for dist in paths.values()
        )

        # 初始化节点状态 - 感染率50%
        model.node_states = {node: SUSCEPTIBLE for node in model.G.nodes}
        initial_infected_cnt = int(model.num_nodes * 0.5)
        initial_infected_nodes = random.sample(list(model.G.nodes), initial_infected_cnt)
        for node in initial_infected_nodes:
            model.node_states[node] = INFECTED

        # 初始化节点状态概率分布
        model.node_probs = {}
        for node in model.G.nodes:
            model.node_probs[node] = {
                SUSCEPTIBLE: 1.0 if model.node_states[node] == SUSCEPTIBLE else 0.0,
                INFECTED: 1.0 if model.node_states[node] == INFECTED else 0.0,
                RECOVERED: 0.0,
                VACCINATED: 0.0
            }

        model.node_list = list(model.G.nodes)
        model.node_idx = {n: i for i, n in enumerate(model.node_list)}

        # 初始化监管者
        model.regulator_positions = random.sample(model.node_list, model.num_regulators)
        model.regulator_visit_history = [defaultdict(int) for _ in range(model.num_regulators)]
        model.regulator_efficiency = [0.0] * model.num_regulators
        model.last_visit_impact = defaultdict(float)
        model.all_visited_nodes = set(model.regulator_positions)

        # 重置结果记录
        model.results = {
            'coverage': [],
            'regulator_I_counts': [],
            'I_to_regulator_distances': [],
            'regulator_target_I_ratios': []
        }

        # 初始化当前步统计变量
        model.regulator_target_I_count_current_step = 0

    @staticmethod
    def run_single_simulation_for_second_row(model):
        """运行第二行专用的单次模拟 - 生成更多样化的数据"""
        SecondRowExperiment.initialize_simulation_with_custom_params(model)

        regulator_target_I_ratios = []

        for step in range(model.max_steps):
            # 重置计数器
            model.regulator_target_I_count_current_step = 0

            # 执行监管者移动
            for i in range(model.num_regulators):
                new_pos = model.regulator_move_based_on_importance(i)
                if new_pos and new_pos in model.G.nodes:
                    model.regulator_positions[i] = new_pos
                    model.all_visited_nodes.add(new_pos)

            # 节点状态转移
            model.mmca_state_transition_with_intervention()

            # 记录监管者目标I节点比例
            if step >= 1:  # 从第2步开始记录
                ratio = model.regulator_target_I_count_current_step / model.num_regulators
                regulator_target_I_ratios.append(ratio)

                # 调试：打印一些步骤的数据
                if step % 10 == 0:
                    I_nodes_count = sum(1 for node in model.G.nodes if model.node_states[node] == INFECTED)
                    print(f"  步骤 {step}: 监管者目标I比例={ratio:.3f}, 当前I节点数={I_nodes_count}")

        return {
            'coverage': model.results['coverage'][-1] if model.results['coverage'] else 0,
            'regulator_target_I_ratios': regulator_target_I_ratios
        }

    @staticmethod
    def run_second_row_experiment(param_range, num_experiments=10):
        """运行第二行专用的delta参数实验 - 只运行三个关键delta值"""
        regulator_target_I_data = [defaultdict(list) for _ in NETWORK_TYPES]

        for param_val in param_range:
            print(f"\n正在计算第二行实验 delta = {param_val:.2f} ...")

            for exp_idx in range(num_experiments):
                # 使用第二行专用参数
                sim_params = SecondRowExperiment.get_base_params()
                sim_params['delta'] = param_val

                for i, (network_type, network_name) in enumerate(NETWORK_TYPES):
                    sim_params['network_type'] = network_type
                    try:
                        sim = RumorDiffusionModel(**sim_params)
                        result = SecondRowExperiment.run_single_simulation_for_second_row(sim)

                        # 收集监管者目标I节点比例数据
                        if result['regulator_target_I_ratios']:
                            regulator_target_I_data[i][param_val].extend(result['regulator_target_I_ratios'])
                            print(
                                f"  {network_name} 实验{exp_idx + 1}: 添加了 {len(result['regulator_target_I_ratios'])} 个数据点")

                    except Exception as e:
                        print(f"第二行实验出错: {e}")
                        continue

        return regulator_target_I_data, param_range


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
            'I_to_regulator_distances': [],
            'regulator_target_I_ratios': []
        }

        # 当前步统计变量
        self.regulator_target_I_count_current_step = 0

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

    def calculate_avg_distance_to_regulators(self, node):
        """计算节点到所有监管者的平均距离（使用预计算结果）"""
        if self.num_regulators == 0:
            return self.num_nodes

        distances = []
        for reg_pos in self.regulator_positions:
            if node in self.all_shortest_paths and reg_pos in self.all_shortest_paths[node]:
                dist = self.all_shortest_paths[node][reg_pos]
                distances.append(max(dist, 1e-6))
            else:
                distances.append(self.num_nodes)
        return np.mean(distances) if distances else self.num_nodes

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
        if random.random() < 0:  # 10%的概率随机选择
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


if __name__ == "__main__":
    print("=== 专门为第二行子图运行的δ实验（使用特殊参数）===")

    # 只运行三个关键delta值：0.1, 0.5, 0.9
    delta_range = [0.1, 0.3, 0.7]

    regulator_target_I_data, delta_range_second_row = SecondRowExperiment.run_second_row_experiment(
        delta_range, NUM_EXPERIMENTS
    )

    # 保存第二行子图数据到txt文件
    data_recorder.save_second_row_data(
        "第二行δ实验",
        regulator_target_I_data,
        [name for _, name in NETWORK_TYPES]
    )

    print("\n=== 第二行实验数据已全部保存完成 ===")
    print(f"数据保存目录: {data_recorder.output_dir}")

    # 打印一些统计信息
    for i, (network_type, network_name) in enumerate(NETWORK_TYPES):
        print(f"\n{network_name}统计信息:")
        total_data_points = 0
        for delta_val, data_list in regulator_target_I_data[i].items():
            if data_list:
                print(f"  Delta={delta_val:.2f}: {len(data_list)}个数据点")
                total_data_points += len(data_list)
        print(f"  总计: {total_data_points}个数据点")
