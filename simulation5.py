import scipy.io as sio
from scipy import sparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------- 全局常量定义 --------------------------
# 节点状态常量
SUSCEPTIBLE = "S"
INFECTED = "I"
RECOVERED = "R"
Vigilant = "Q"

# 模拟核心参数
BA_M = 15
NUM_NODES = 500
# 调整监管者数量对比组，增加差异
REGULATOR_COUNTS = [0, 60, 120]  # 减少监管者数量以避免索引问题
MAX_STEPS = 35  # 迭代步数
INITIAL_INFECTED_RATIO = 0.02
NUM_REPEATS = 20  # 减少重复次数以加快调试
# 均衡智能策略
BALANCED_STRATEGY = (0.3, 0.4, 0.45, "均衡智能策略")

# 显著增大β参数组合的差异
BETA_COMBINATIONS = [
    (0.12, 0.04, "β1=0.08, β2=0.03"),  # 低传播力
    (0.25, 0.1, "β1=0.25, β2=0.1"),  # 中等传播力
    (0.6, 0.25, "β1=0.6, β2=0.25")  # 高传播力
]

# 绘图样式配置 - 所有线条为实线，定义不同标记样式
STATE_STYLES = {
    'S': {'color': '#2E86AB', 'linestyle': '-', 'label': '易感(S)', 'marker': 'o'},
    'I': {'color': '#A23B72', 'linestyle': '-', 'label': '感染(I)', 'marker': 's'},
    'Q': {'color': '#F18F01', 'linestyle': '-', 'label': '警惕(Q)', 'marker': '^'},
    'R': {'color': '#C73E1D', 'linestyle': '-', 'label': '恢复(R)', 'marker': 'D'}
}

# 计数点数量
NUM_MARKER_POINTS = 20

# 并行运行重复实验的线程数；None 表示按 CPU 核心数和 NUM_REPEATS 自动选择
MAX_WORKERS = None


def load_caltech_facebook_graph(mtx_file_path):
    """加载加州理工Facebook网络的MTX文件 - 更健壮的版本"""
    print(f"正在加载MTX文件: {mtx_file_path}")

    try:
        # 方法1: 尝试使用scipy读取MTX文件
        print("尝试使用scipy.io.mmread读取...")
        matrix = sio.mmread(mtx_file_path)

        # 转换为networkx图
        if sparse.isspmatrix(matrix):
            # 如果是稀疏矩阵，转换为coo格式
            matrix = matrix.tocoo()
            G = nx.Graph()

            # 添加边
            for i, j, v in zip(matrix.row, matrix.col, matrix.data):
                if i != j:  # 避免自环
                    G.add_edge(int(i), int(j))

            print(f"成功加载图: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
            return G

    except Exception as e:
        print(f"使用scipy读取失败: {e}")

        # 方法2: 手动解析MTX文件
        print("尝试手动解析MTX文件...")
        try:
            G = nx.Graph()
            node_set = set()

            with open(mtx_file_path, 'r') as f:
                for line in f:
                    # 跳过注释行
                    if line.startswith('%'):
                        continue

                    # 分割行
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    try:
                        # 尝试解析为整数
                        u = int(parts[0])
                        v = int(parts[1])

                        # 添加到图中
                        G.add_edge(u, v)
                        node_set.add(u)
                        node_set.add(v)

                    except ValueError:
                        # 如果解析失败，跳过这一行
                        continue

            print(f"手动解析成功: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")

            # 如果图为空，尝试其他方法
            if G.number_of_nodes() == 0:
                raise ValueError("手动解析后图为空")

            return G

        except Exception as e2:
            print(f"手动解析也失败: {e2}")

            # 方法3: 使用networkx的edgelist读取，但不处理边数据
            print("尝试使用networkx.read_edgelist（简化版）...")
            try:
                G = nx.read_edgelist(
                    mtx_file_path,
                    comments='%',
                    nodetype=int,
                    data=False  # 不尝试解析边数据
                )
                print(f"使用edgelist读取成功: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
                return G
            except Exception as e3:
                print(f"所有方法都失败: {e3}")
                raise

    return None


def normalize_graph_nodes(G):
    """将图的节点重新映射为从0开始的连续整数"""
    if G is None:
        return None

    # 获取原始节点列表并排序
    original_nodes = sorted(G.nodes())

    # 创建节点映射
    node_mapping = {old_node: new_index for new_index, old_node in enumerate(original_nodes)}

    # 创建新图并重新映射节点
    normalized_G = nx.Graph()
    for u, v in G.edges():
        normalized_G.add_edge(node_mapping[u], node_mapping[v])

    print(f"节点重新映射完成: {len(original_nodes)}个节点已重新映射为0-{len(original_nodes) - 1}")
    return normalized_G


class SIVRSimulation:
    def __init__(self, num_nodes=NUM_NODES, avg_degree=BA_M, num_regulators=30,
                 beta1=0.2, beta2=0.05,
                 mu=0.08,  # 增大I→R的转移系数，加快恢复
                 gamma1=0.3,  # 增大V→S的转移系数，免疫更容易失效
                 gamma2=0.18,  # 增大R→V的转移系数，更快转化为免疫
                 xi=0.6,  # 增大对抗效应系数，增强监管效果
                 max_steps=MAX_STEPS,
                 w_topology=0.2, w_risk=0.3, w_spread=0.3,
                 mmax=15, delta=0.85,  # 调整战略权重参数δ
                 graph=None):  # 新增：允许传入外部图
        """SIVR模型初始化：整合改进的谣言传播与监管策略"""
        # 状态映射
        self.STATE = {
            'S': 0,  # 易感状态
            'Q': 1,  # 免疫状态
            'I': 2,  # 感染状态
            'R': 3  # 恢复状态
        }

        # 基础网络参数
        self.num_nodes = num_nodes
        self.ba_m = avg_degree  # BA网络参数
        self.num_regulators = num_regulators  # 监管者数量
        self.max_steps = max_steps
        self.external_graph = graph  # 存储外部传入的图

        # 模型核心参数 - 调整以增强状态差异
        self.beta1 = beta1  # S→I传播系数
        self.beta2 = beta2  # V→I突破系数
        self.mu = mu  # I→R恢复系数
        self.gamma1 = gamma1  # V→S失效率
        self.gamma2 = gamma2  # R→V转化率
        self.xi = xi  # 对抗因子系数
        self.delta = delta  # 战略权重参数
        self.zhijing = None

        # 监管策略参数
        self.w_topology = w_topology  # 拓扑权重
        self.w_risk = w_risk  # 风险权重
        self.w_spread = w_spread  # 传播权重
        self.mmax = mmax

        # 网络与节点特征
        self.all_shortest_paths = None

        # 节点状态与概率
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
            'S': [], 'I': [], 'Q': [], 'R': [],
            'regulator_efficiency': []
        }

    def initialize_simulation(self):
        """初始化模拟环境 - 支持外部图或BA网络"""
        # 如果提供了外部图，使用外部图；否则创建BA网络
        if self.external_graph is not None:
            # 确保节点编号是连续的
            self.G = normalize_graph_nodes(self.external_graph)
            self.num_nodes = len(self.G.nodes())
            print(f"使用外部图，节点数量: {self.num_nodes}")
        else:
            # 创建连通BA网络
            self.G = nx.barabasi_albert_graph(self.num_nodes, self.ba_m)
            while not nx.is_connected(self.G):
                self.G = nx.barabasi_albert_graph(self.num_nodes, self.ba_m)
            print(f"生成BA网络，节点数量: {self.num_nodes}")

        # 确保网络连通
        if not nx.is_connected(self.G):
            print("警告: 网络不连通，使用最大连通分量")
            largest_cc = max(nx.connected_components(self.G), key=len)
            self.G = self.G.subgraph(largest_cc).copy()
            self.num_nodes = len(self.G.nodes())
            print(f"使用最大连通分量，节点数量: {self.num_nodes}")

            # 重新规范化节点编号
            self.G = normalize_graph_nodes(self.G)

        # 计算节点拓扑特征
        print("正在预计算最短路径...这可能需要一些时间")
        try:
            self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
            print("最短路径预计算完成")

            # 计算网络直径
            if self.all_shortest_paths:
                self.zhijing = max(
                    dist for paths in self.all_shortest_paths.values()
                    for dist in paths.values()
                )
            else:
                self.zhijing = 0
        except Exception as e:
            print(f"最短路径计算失败: {e}")
            print("使用近似直径计算...")
            try:
                self.zhijing = nx.diameter(self.G)
            except:
                self.zhijing = 10  # 默认值
            self.all_shortest_paths = {}

        # 初始化节点状态 - 使用安全的索引方式
        self.states = np.full(self.num_nodes, self.STATE['S'], dtype=int)
        initial_infected_cnt = max(1, int(self.num_nodes * INITIAL_INFECTED_RATIO))

        # 确保选择的节点索引在范围内
        available_nodes = list(range(self.num_nodes))
        initial_infected_nodes = random.sample(available_nodes, min(initial_infected_cnt, len(available_nodes)))

        for node in initial_infected_nodes:
            if node < self.num_nodes:  # 安全检查
                self.states[node] = self.STATE['I']

        # 初始化节点状态概率分布
        self.node_probs = {}
        for node in range(self.num_nodes):  # 使用数字索引而不是节点对象
            self.node_probs[node] = np.zeros(4)
            if self.states[node] == self.STATE['S']:
                self.node_probs[node][self.STATE['S']] = 1.0
            elif self.states[node] == self.STATE['I']:
                self.node_probs[node][self.STATE['I']] = 1.0
            elif self.states[node] == self.STATE['R']:
                self.node_probs[node][self.STATE['R']] = 1.0
            elif self.states[node] == self.STATE['Q']:
                self.node_probs[node][self.STATE['Q']] = 1.0

        # 修复：更安全的监管者初始化
        self.regulator_positions = []
        if self.num_regulators > 0:
            # 获取所有感染节点
            infected_nodes = [node for node in range(self.num_nodes) if self.states[node] == self.STATE['I']]

            if not infected_nodes:
                # 如果没有感染节点，随机选择一些节点
                self.regulator_positions = random.sample(list(range(self.num_nodes)),
                                                         min(self.num_regulators, self.num_nodes))
            else:
                # 确保不超过感染节点数量
                num_to_select = min(self.num_regulators, len(infected_nodes))
                self.regulator_positions = random.sample(infected_nodes, num_to_select)

                # 如果还需要更多监管者，从其他节点中随机选择
                if len(self.regulator_positions) < self.num_regulators:
                    remaining_nodes = [node for node in range(self.num_nodes) if node not in self.regulator_positions]
                    additional_count = min(self.num_regulators - len(self.regulator_positions), len(remaining_nodes))
                    additional_positions = random.sample(remaining_nodes, additional_count)
                    self.regulator_positions.extend(additional_positions)

            print(f"初始监管者部署: {len(self.regulator_positions)}个监管者")

            # 初始化监管者历史记录
            self.regulator_visit_history = [defaultdict(int) for _ in range(len(self.regulator_positions))]
            self.regulator_efficiency = [0.0] * len(self.regulator_positions)
        else:
            self.regulator_visit_history = []
            self.regulator_efficiency = [0.0]

        # 初始化其他记录
        self.last_visit_impact.clear()
        self.last_visit_time.clear()
        for node in range(self.num_nodes):
            self.last_visit_time[node] = -1000

        # 记录初始状态
        self.record_state_density()

    def record_state_density(self):
        """记录当前各状态密度"""
        total = self.num_nodes
        s_density = np.sum(self.states == self.STATE['S']) / total
        i_density = np.sum(self.states == self.STATE['I']) / total
        v_density = np.sum(self.states == self.STATE['Q']) / total
        r_density = np.sum(self.states == self.STATE['R']) / total

        self.results['S'].append(s_density)
        self.results['I'].append(i_density)
        self.results['Q'].append(v_density)
        self.results['R'].append(r_density)

        if len(self.results['regulator_efficiency']) == 0:
            self.results['regulator_efficiency'].append(0.0)

    def calculate_avg_distance_to_regulators(self, node):
        """计算节点到所有监管者的平均距离（使用预计算结果）"""
        if len(self.regulator_positions) == 0:
            return self.num_nodes  # 无监管者时返回节点总数作为默认距离

        distances = []
        for reg_pos in self.regulator_positions:
            # 使用预计算的最短路径结果
            if reg_pos in self.all_shortest_paths and node in self.all_shortest_paths[reg_pos]:
                dist = self.all_shortest_paths[reg_pos][node]
                distances.append(max(dist, 1e-6))
            else:
                # 如果预计算中没有找到路径，使用网络直径作为估计
                distances.append(self.zhijing if self.zhijing > 0 else self.num_nodes)
        return np.mean(distances) if distances else self.num_nodes

    def calculate_confrontation_factor(self, node):
        """计算对抗因子（指数衰减）"""
        avg_dist = self.calculate_avg_distance_to_regulators(node)
        return min(1,self.xi / (1 + np.exp(-avg_dist)))

    def calculate_propagation_importance(self, node_j, current_step):
        """计算节点j的传播重要性Ψ_j(t) - 公式(6)"""
        if node_j >= self.num_nodes:  # 安全检查
            return 0.0

        if self.states[node_j] == self.STATE['S']:
            # S状态节点：预防重要性
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                if neighbor_k >= self.num_nodes:  # 安全检查
                    continue
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_I = self.node_probs[neighbor_k][self.STATE['I']]
                importance += c_j * self.beta1 * p_k_I * 1.0  # a_jk = 1.0 (存在连接)
            return (1 - self.delta) * importance

        elif self.states[node_j] == self.STATE['I']:
            # I状态节点：干预重要性
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                if neighbor_k >= self.num_nodes:  # 安全检查
                    continue
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_S = self.node_probs[neighbor_k][self.STATE['S']]
                p_k_V = self.node_probs[neighbor_k][self.STATE['Q']]
                term = c_j * self.beta1 * p_k_S + c_j * self.beta2 * p_k_V
                importance += term * 1.0  # a_jk = 1.0
            return self.delta * importance

        else:
            # 其他状态节点重要性为0
            return 0.0

    def regulator_next_move(self, regulator_idx):
        """监管者移动决策 - 基于传播重要性的随机游走 - 公式(7)"""
        # 修复：添加索引检查
        if regulator_idx >= len(self.regulator_positions):
            print(f"警告: 监管者索引 {regulator_idx} 超出范围 (0-{len(self.regulator_positions) - 1})")
            # 返回一个随机有效位置
            if self.regulator_positions:
                return random.choice(list(range(self.num_nodes)))
            else:
                return 0

        current_pos = self.regulator_positions[regulator_idx]

        # 获取当前节点的邻居
        try:
            neighbors = list(self.G.neighbors(current_pos))
        except:
            neighbors = []

        if not neighbors:
            return current_pos  # 如果没有邻居，保持原地

        # 计算每个邻居的传播重要性
        psi_values = []
        for neighbor in neighbors:
            if neighbor < self.num_nodes:  # 安全检查
                psi = self.calculate_propagation_importance(neighbor, len(self.results['S']) - 1)
                psi_values.append(psi)
            else:
                psi_values.append(0.0)

        # 计算转移概率 - 公式(7)
        total_psi = sum(psi_values)
        if total_psi == 0:
            # 如果所有邻居重要性都为0，均匀随机选择
            transition_probs = [1.0 / len(neighbors)] * len(neighbors)
        else:
            transition_probs = [psi / total_psi for psi in psi_values]

        # 根据概率选择下一个位置
        try:
            next_pos = np.random.choice(neighbors, p=transition_probs)
            return next_pos
        except ValueError:
            # 概率和不为1时的容错处理
            return random.choice(neighbors)

    def calculate_intervention_probability(self, node_i):
        """计算节点i的干预概率ρ_i(t) - 公式(8)"""
        if len(self.regulator_positions) == 0:
            return 0.0

        prob_no_regulator = 1.0
        for regulator_pos in self.regulator_positions:
            if regulator_pos == node_i:
                # 如果监管者在节点i，概率为1
                prob_no_regulator *= 0.0
                break

        return 1.0 - prob_no_regulator

    def execute_regulatory_action(self, node):
        """执行监管动作 - 与理论完全一致"""
        if node >= self.num_nodes:  # 安全检查
            return 0.0

        original_state = self.states[node]

        if original_state == self.STATE['S']:
            # S → V
            self.states[node] = self.STATE['Q']
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE['Q']] = 1.0
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
        """计算感染概率（q_i和v_i）- 公式(11)"""
        if node >= self.num_nodes:  # 安全检查
            return 0.0, 0.0

        try:
            neighbors = list(self.G.neighbors(node))
        except:
            neighbors = []

        if not neighbors:
            return 0.0, 0.0

        c_i = self.calculate_confrontation_factor(node)
        product_q = 1.0
        product_v = 1.0

        for neighbor in neighbors:
            if neighbor >= self.num_nodes:  # 安全检查
                continue
            a_ji = 1.0
            p_j_I = self.node_probs[neighbor][self.STATE['I']]
            product_q *= (1 - a_ji * c_i * self.beta1 * p_j_I)
            product_v *= (1 - a_ji * c_i * self.beta2 * p_j_I)

        q_i = 1 - product_q
        v_i = 1 - product_v
        return max(0.0, min(1.0, q_i)), max(0.0, min(1.0, v_i))

    def update_regulators(self):
        """更新监管者位置并执行监管动作 - 与理论一致"""
        if len(self.regulator_positions) == 0:
            self.results['regulator_efficiency'].append(0.0)
            return

        total_efficiency = 0
        current_step = len(self.results['S']) - 1

        # 更新监管者位置 - 公式(9)
        new_positions = []
        for i in range(len(self.regulator_positions)):  # 修复：使用实际监管者数量
            new_pos = self.regulator_next_move(i)
            # 确保新位置在有效范围内
            if new_pos >= self.num_nodes:
                new_pos = random.randint(0, self.num_nodes - 1)
            new_positions.append(new_pos)

            # 记录访问历史
            if i < len(self.regulator_visit_history):  # 添加边界检查
                self.regulator_visit_history[i][new_pos] = current_step
            self.last_visit_time[new_pos] = current_step

            # 计算移动效率（基于目标节点的重要性）
            psi_target = self.calculate_propagation_importance(new_pos, current_step)
            if i < len(self.regulator_efficiency):  # 添加边界检查
                self.regulator_efficiency[i] = psi_target
            total_efficiency += psi_target

        self.regulator_positions = new_positions

        # 执行监管干预
        intervention_impact = 0
        for i, pos in enumerate(self.regulator_positions):
            impact = self.execute_regulatory_action(pos)
            intervention_impact += impact

        # 记录平均监管效率
        avg_efficiency = total_efficiency / len(self.regulator_positions) if self.regulator_positions else 0.0
        self.results['regulator_efficiency'].append(avg_efficiency)

    def update_mmca_states(self):
        """更新所有节点的状态（整合监管干预的MMCA机制）- 公式(10)"""
        # 计算每个节点的干预概率
        intervention_probs = {}
        for node in range(self.num_nodes):
            intervention_probs[node] = self.calculate_intervention_probability(node)

        for node in range(self.num_nodes):
            rho_i = intervention_probs[node]
            current_state = self.states[node]

            # 计算感染概率
            q_i, v_i = self.calculate_infection_probabilities(node)

            # 根据当前状态和干预概率更新状态概率
            if current_state == self.STATE['S']:
                # S → I (感染) 或 S → V (监管干预)
                if np.random.random() < rho_i:
                    # 被监管干预：S → V
                    self.states[node] = self.STATE['Q']
                    self.node_probs[node][self.STATE['Q']] = 1.0
                else:
                    # 正常感染过程
                    if np.random.random() < q_i:
                        self.states[node] = self.STATE['I']
                        self.node_probs[node][self.STATE['I']] = 1.0
                    else:
                        self.states[node] = self.STATE['S']
                        self.node_probs[node][self.STATE['S']] = 1.0

            elif current_state == self.STATE['Q']:
                # V → S (失效) 或 V → I (突破感染)
                if np.random.random() < self.gamma1:
                    self.states[node] = self.STATE['S']
                    self.node_probs[node][self.STATE['S']] = 1.0
                elif np.random.random() < v_i:
                    self.states[node] = self.STATE['I']
                    self.node_probs[node][self.STATE['I']] = 1.0
                else:
                    self.states[node] = self.STATE['Q']
                    self.node_probs[node][self.STATE['Q']] = 1.0

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
                # R → Q (转化为免疫)
                if np.random.random() < self.gamma2:
                    self.states[node] = self.STATE['Q']
                    self.node_probs[node][self.STATE['Q']] = 1.0
                else:
                    self.states[node] = self.STATE['R']
                    self.node_probs[node][self.STATE['R']] = 1.0

    def step(self):
        """执行一步模拟 - 调整顺序以符合理论"""
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
            if step % 10 == 0:  # 每10步显示一次进度
                print(f"  模拟进度: {step + 1}/{self.max_steps}")
            self.step()
        return self.results


def run_single_repeat(simulation_kwargs):
    """运行一次独立重复实验，供线程池调用。"""
    simulator = SIVRSimulation(**simulation_kwargs)
    return simulator.run_simulation()


def run_repeats_parallel(simulation_kwargs, repeat_count=NUM_REPEATS, max_workers=MAX_WORKERS):
    """多线程运行重复实验，并把结果累加。"""
    if max_workers is None:
        max_workers = min(repeat_count, os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, repeat_count))

    sum_results = {
        'S': np.zeros(MAX_STEPS),
        'I': np.zeros(MAX_STEPS),
        'Q': np.zeros(MAX_STEPS),
        'R': np.zeros(MAX_STEPS)
    }

    print(f"  使用 {max_workers} 个线程并行运行 {repeat_count} 次重复实验...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_repeat, simulation_kwargs.copy())
            for _ in range(repeat_count)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            results = future.result()
            for state in ['S', 'I', 'Q', 'R']:
                sum_results[state] += np.array(results[state])

            if completed % 5 == 0 or completed == repeat_count:
                print(f"  已完成 {completed}/{repeat_count} 次实验")

    return sum_results


def save_all_results_to_single_file(all_results, filename, base_params, strategy_label):
    """将所有实验数据存储到单个TXT文件"""
    # 创建结果目录
    if not os.path.exists("simulation_results"):
        os.makedirs("simulation_results")

    filepath = os.path.join("simulation_results", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        # 写入全局参数信息
        f.write("=" * 80 + "\n")
        f.write("SIVR谣言传播模拟实验综合数据\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("=== 全局实验参数 ===\n")
        f.write(f"策略名称: {strategy_label}\n")
        for key, value in base_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"重复实验次数: {NUM_REPEATS}\n")
        f.write(f"总时间步数: {MAX_STEPS}\n\n")

        # 写入β参数组合实验数据
        f.write("=" * 80 + "\n")
        f.write("第一部分: 不同β参数组合实验结果\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results['beta_experiments']:
            f.write(f"--- {result['label']} ---\n")
            f.write("参数详情:\n")
            for key, value in result['params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n数据格式: 时间步\tS密度\tI密度\tV密度\tR密度\n")
            for step in range(len(result['S'])):
                f.write(f"{step}\t{result['S'][step]:.6f}\t{result['I'][step]:.6f}\t"
                        f"{result['Q'][step]:.6f}\t{result['R'][step]:.6f}\n")
            f.write("\n" + "-" * 60 + "\n\n")

        # 写入监管者数量实验数据
        f.write("=" * 80 + "\n")
        f.write("第二部分: 不同监管者数量实验结果\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results['regulator_experiments']:
            f.write(f"--- {result['label']} ---\n")
            f.write("参数详情:\n")
            for key, value in result['params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n数据格式: 时间步\tS密度\tI密度\tV密度\tR密度\n")
            for step in range(len(result['S'])):
                f.write(f"{step}\t{result['S'][step]:.6f}\t{result['I'][step]:.6f}\t"
                        f"{result['Q'][step]:.6f}\t{result['R'][step]:.6f}\n")
            f.write("\n" + "-" * 60 + "\n\n")

    print(f"所有实验数据已保存至: {filepath}")


def get_marker_indices(total_steps, num_points):
    """计算要显示标记点的索引位置"""
    if num_points >= total_steps:
        return list(range(total_steps))
    # 计算均匀分布的索引
    indices = np.linspace(0, total_steps - 1, num_points, dtype=int)
    # 确保包含起点和终点
    indices[0] = 0
    indices[-1] = total_steps - 1
    # 去重并排序
    return sorted(list(set(indices)))


def run_experiments_and_plot(use_real_network=False, mtx_file_path=None):
    """运行所有实验并生成综合对比图"""

    # 加载真实网络（如果指定）
    real_graph = None
    if use_real_network and mtx_file_path:
        print("使用真实网络数据...")
        real_graph = load_caltech_facebook_graph(mtx_file_path)
        if real_graph is None:
            print("无法加载真实网络，使用BA网络")
            use_real_network = False
        else:
            # 规范化节点编号
            real_graph = normalize_graph_nodes(real_graph)

    # 模拟参数（使用均衡策略）
    w_top, w_risk, w_spread, strategy_label = BALANCED_STRATEGY

    # 根据是否使用真实网络调整参数
    if use_real_network:
        num_nodes_real = real_graph.number_of_nodes()
        base_params = {
            '节点数量': num_nodes_real,
            '网络类型': '加州理工Facebook网络',
            '恢复系数(μ)': 0.08,
            'Q→S失效率(γ₁)': 0.03,
            'R→Q转化率(γ₂)': 0.18,
            '对抗效应系数(ξ)': 0.6,
            '拓扑权重': w_top,
            '风险权重': w_risk,
            '传播权重': w_spread,
            '战略权重(δ)': 0.85
        }
    else:
        base_params = {
            '节点数量': NUM_NODES,
            'BA网络参数(m)': BA_M,
            '恢复系数(μ)': 0.08,
            'Q→S失效率(γ₁)': 0.03,
            'R→Q转化率(γ₂)': 0.18,
            '对抗效应系数(ξ)': 0.6,
            '拓扑权重': w_top,
            '风险权重': w_risk,
            '传播权重': w_spread,
            '战略权重(δ)': 0.85
        }

    # 存储所有实验结果
    all_results = {
        'beta_experiments': [],
        'regulator_experiments': []
    }

    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 网络类型标签
    network_label = "real" if use_real_network else "synthetic"

    # 1. 运行不同β参数组合的实验 - 监管者数量固定为50个
    for beta1, beta2, label in BETA_COMBINATIONS:
        network_desc = "真实网络" if use_real_network else "BA网络"
        print(f"运行 {label}，网络={network_desc}，监管者数量=50，共{NUM_REPEATS}次重复实验...")

        # 初始化累加器
        sum_results = {
            'S': np.zeros(MAX_STEPS),
            'I': np.zeros(MAX_STEPS),
            'Q': np.zeros(MAX_STEPS),
            'R': np.zeros(MAX_STEPS)
        }

        # 多次重复实验
        for repeat in range(NUM_REPEATS):
            if (repeat + 1) % 5 == 0:
                print(f"  已完成 {repeat + 1}/{NUM_REPEATS} 次实验")

            # 单次实验 - 监管者数量固定为50
            simulator = SIVRSimulation(
                beta1=beta1,
                beta2=beta2,
                num_regulators=50,  # 修改：监管者数量固定为50
                graph=real_graph if use_real_network else None,  # 传入真实网络
                num_nodes=base_params['节点数量'],
                avg_degree=base_params.get('BA网络参数(m)', BA_M),
                mu=base_params['恢复系数(μ)'],
                gamma1=base_params['Q→S失效率(γ₁)'],
                gamma2=base_params['R→Q转化率(γ₂)'],
                xi=base_params['对抗效应系数(ξ)'],
                w_topology=base_params['拓扑权重'],
                w_risk=base_params['风险权重'],
                w_spread=base_params['传播权重'],
                delta=base_params['战略权重(δ)']
            )
            results = simulator.run_simulation()

            # 累加结果
            for state in ['S', 'I', 'Q', 'R']:
                sum_results[state] += np.array(results[state])

        # 计算平均值
        avg_results = {
            'label': f"{label} ({network_desc}, 监管者=50)",
            'params': {
                'β1值': beta1,
                'β2值': beta2,
                '监管者数量': 50,  # 修改：明确显示监管者数量为50
                '战略权重δ': base_params['战略权重(δ)'],
                '网络类型': network_desc
            },
            'S': sum_results['S'] / NUM_REPEATS,
            'I': sum_results['I'] / NUM_REPEATS,
            'Q': sum_results['Q'] / NUM_REPEATS,
            'R': sum_results['R'] / NUM_REPEATS
        }
        all_results['beta_experiments'].append(avg_results)

    # 2. 运行不同监管者数量的实验 - β参数固定为β₁=0.15, β₂=0.05
    fixed_beta1 = 0.15  # 修改：固定β₁=0.15
    fixed_beta2 = 0.05  # 修改：固定β₂=0.05
    beta_label = f"β1={fixed_beta1}, β2={fixed_beta2}"

    for reg_count in REGULATOR_COUNTS:
        network_desc = "真实网络" if use_real_network else "BA网络"
        print(f"运行 {beta_label}，网络={network_desc}，监管者数量={reg_count}，共{NUM_REPEATS}次重复实验...")

        # 初始化累加器
        sum_results = {
            'S': np.zeros(MAX_STEPS),
            'I': np.zeros(MAX_STEPS),
            'Q': np.zeros(MAX_STEPS),
            'R': np.zeros(MAX_STEPS)
        }

        # 多次重复实验
        for repeat in range(NUM_REPEATS):
            if (repeat + 1) % 5 == 0:
                print(f"  已完成 {repeat + 1}/{NUM_REPEATS} 次实验")

            # 单次实验 - β参数固定
            simulator = SIVRSimulation(
                beta1=fixed_beta1,  # 修改：使用固定的β₁
                beta2=fixed_beta2,  # 修改：使用固定的β₂
                num_regulators=reg_count,
                graph=real_graph if use_real_network else None,  # 传入真实网络
                num_nodes=base_params['节点数量'],
                avg_degree=base_params.get('BA网络参数(m)', BA_M),
                mu=base_params['恢复系数(μ)'],
                gamma1=base_params['Q→S失效率(γ₁)'],
                gamma2=base_params['R→Q转化率(γ₂)'],
                xi=base_params['对抗效应系数(ξ)'],
                w_topology=base_params['拓扑权重'],
                w_risk=base_params['风险权重'],
                w_spread=base_params['传播权重'],
                delta=base_params['战略权重(δ)']
            )
            results = simulator.run_simulation()

            # 累加结果
            for state in ['S', 'I', 'Q', 'R']:
                sum_results[state] += np.array(results[state])

        # 计算平均值
        avg_results = {
            'label': f"{beta_label}, {network_desc}, 监管者={reg_count}",
            'params': {
                'β1值': fixed_beta1,
                'β2值': fixed_beta2,
                '监管者数量': reg_count,
                '战略权重δ': base_params['战略权重(δ)'],
                '网络类型': network_desc
            },
            'S': sum_results['S'] / NUM_REPEATS,
            'I': sum_results['I'] / NUM_REPEATS,
            'Q': sum_results['Q'] / NUM_REPEATS,
            'R': sum_results['R'] / NUM_REPEATS
        }
        all_results['regulator_experiments'].append(avg_results)

    # 保存所有结果到单个文件
    save_all_results_to_single_file(
        all_results,
        f"sivr_{network_label}_network_results_{timestamp}.txt",
        base_params,
        strategy_label
    )

    # 计算要显示的标记点索引
    marker_indices = get_marker_indices(MAX_STEPS, NUM_MARKER_POINTS)

    # 创建包含6个子图的综合对比图
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    network_type = "真实网络" if use_real_network else "合成网络"
    fig.suptitle(f'{strategy_label}在{network_type}上的状态演化（{NUM_REPEATS}次实验均值）', fontsize=18)

    # 绘制β参数组合的子图（上3个子图）- 监管者数量固定为50
    for i, result in enumerate(all_results['beta_experiments']):
        ax = axes[i, 0]
        for state in ['S', 'I', 'Q', 'R']:
            # 绘制实线
            ax.plot(
                range(MAX_STEPS),
                result[state],
                color=STATE_STYLES[state]['color'],
                linestyle=STATE_STYLES[state]['linestyle'],
                linewidth=2,
                label=STATE_STYLES[state]['label']
            )
            # 添加计数点
            ax.scatter(
                [step for step in marker_indices],
                [result[state][step] for step in marker_indices],
                color=STATE_STYLES[state]['color'],
                marker=STATE_STYLES[state]['marker'],
                s=40,  # 点的大小
                zorder=3  # 确保点在实线上方
            )

        ax.set_title(f'{result["label"]}', fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_xlim(0, MAX_STEPS - 1)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)

    # 绘制监管者数量的子图（下3个子图）- β参数固定
    for i, result in enumerate(all_results['regulator_experiments']):
        ax = axes[i, 1]
        for state in ['S', 'I', 'Q', 'R']:
            # 绘制实线
            ax.plot(
                range(MAX_STEPS),
                result[state],
                color=STATE_STYLES[state]['color'],
                linestyle=STATE_STYLES[state]['linestyle'],
                linewidth=2,
                label=STATE_STYLES[state]['label']
            )
            # 添加计数点
            ax.scatter(
                [step for step in marker_indices],
                [result[state][step] for step in marker_indices],
                color=STATE_STYLES[state]['color'],
                marker=STATE_STYLES[state]['marker'],
                s=40,  # 点的大小
                zorder=3  # 确保点在实线上方
            )

        ax.set_title(f'{result["label"]}', fontsize=14)
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('密度', fontsize=12)
        ax.set_xlim(0, MAX_STEPS - 1)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图表
    if not os.path.exists("comparison_results"):
        os.makedirs("comparison_results")
    plt.savefig(f"comparison_results/{network_label}_network_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"{network_type}结果图表已保存至 comparison_results 目录")

    # 显示图表
    plt.show()

    # 输出最终统计
    network_desc = "真实网络" if use_real_network else "合成网络"
    print(f"\n=== β参数组合在{network_desc}上的最终效果统计（监管者=50，{NUM_REPEATS}次实验均值） ===")
    for result in all_results['beta_experiments']:
        print(f"{result['label']}:")
        print(f"  易感密度: {result['S'][-1]:.4f}")
        print(f"  感染密度: {result['I'][-1]:.4f}")
        print(f"  免疫密度: {result['Q'][-1]:.4f}")
        print(f"  恢复密度: {result['R'][-1]:.4f}\n")

    print(f"\n=== 监管者数量在{network_desc}上的最终效果统计（β1=0.15, β2=0.05，{NUM_REPEATS}次实验均值） ===")
    for result in all_results['regulator_experiments']:
        print(f"{result['label']}:")
        print(f"  易感密度: {result['S'][-1]:.4f}")
        print(f"  感染密度: {result['I'][-1]:.4f}")
        print(f"  免疫密度: {result['Q'][-1]:.4f}")
        print(f"  恢复密度: {result['R'][-1]:.4f}\n")


if __name__ == "__main__":
    # 使用真实网络数据
    mtx_file_path = r"D:\Users\lenovo\Desktop\文件\比赛科研\论文\我的\Rumor Propagation and Supervision during Confrontation An\加州理工数据\socfb-Caltech36.mtx"

    # 检查文件是否存在
    if os.path.exists(mtx_file_path):
        print(f"找到MTX文件: {mtx_file_path}")
        # 运行真实网络实验
        run_experiments_and_plot(use_real_network=True, mtx_file_path=mtx_file_path)

        # 也可以同时运行合成网络实验进行对比
        print("\n" + "=" * 50)
        print("现在运行合成网络实验进行对比...")
        print("=" * 50)
        run_experiments_and_plot(use_real_network=False)
    else:
        print(f"未找到MTX文件: {mtx_file_path}")
        print("请检查文件路径，将运行合成网络实验...")
        run_experiments_and_plot(use_real_network=False)
