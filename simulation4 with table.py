import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import random
import seaborn as sns
from scipy import stats
import warnings
import os
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

warnings.filterwarnings('ignore')

# 定义节点状态常量
SUSCEPTIBLE = "S"
INFECTED = "I"
RECOVERED = "R"
VACCINATED = "V"

# BA网络参数
BA_M = 6

# 并行运行重复实验的进程数；None 表示使用 min(重复次数, CPU核心数)
MAX_WORKERS = None


class SIVRRumorSimulation:
    """完整的SIVR谣言传播模拟器 - 基于传播重要性的监管版本"""

    def __init__(self, num_nodes=500, num_regulators=6, beta1=0.3, beta2=0.15,
                 mu=0.08, gamma1=0.1, gamma2=0.05, max_steps=60, xi=0.5, mmax=15,
                 delta=0.8):  # 新增delta参数
        self.regulator_efficiency = [0] * num_regulators if num_regulators > 0 else []
        self.num_nodes = num_nodes
        self.num_regulators = num_regulators
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.max_steps = max_steps
        self.xi = xi
        self.mmax = mmax
        self.delta = delta  # 策略权重参数δ ∈ [0,1]

        # 网络和状态
        self.G = None
        self.node_states = None
        self.node_degrees = None
        self.zhijing = None
        self.regulator_positions = []
        self.all_shortest_paths = None
        self.node_probs = None

        # 监管者历史记录
        self.regulator_visit_history = []
        self.last_visit_impact = defaultdict(float)
        self.last_visit_time = defaultdict(int)

        # 结果记录
        self.results = {
            'S': [], 'I': [], 'V': [], 'R': [],
            'regulator_efficiency': []
        }

        # 状态映射
        self.STATE_MAP = {SUSCEPTIBLE: 0, VACCINATED: 1, INFECTED: 2, RECOVERED: 3}
        self.STATE = {
            'S': SUSCEPTIBLE,
            'V': VACCINATED,
            'I': INFECTED,
            'R': RECOVERED
        }

    def calculate_propagation_importance(self, node_j, current_step):
        """
        计算节点j的传播重要性Ψ_j(t) - 根据理论公式(6)

        Args:
            node_j: 目标节点
            current_step: 当前时间步

        Returns:
            Ψ_j(t): 传播重要性得分
        """
        if self.node_states[node_j] == SUSCEPTIBLE:
            # S状态节点: Ψ_j(t) = (1-δ) * Σ_{k∈N(j)} [c_j(t) * β1 * p_k^I(t) * a_jk]
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_I = self.node_probs[neighbor_k][self.STATE_MAP[INFECTED]]
                a_jk = 1.0  # 邻接矩阵元素
                importance += c_j * self.beta1 * p_k_I * a_jk
            return (1 - self.delta) * importance

        elif self.node_states[node_j] == INFECTED:
            # I状态节点: Ψ_j(t) = δ * Σ_{k∈N(j)} [c_j(t) * β1 * p_k^S(t) + c_j(t) * β2 * p_k^V(t)] * a_jk
            importance = 0.0
            for neighbor_k in self.G.neighbors(node_j):
                c_j = self.calculate_confrontation_factor(node_j)
                p_k_S = self.node_probs[neighbor_k][self.STATE_MAP[SUSCEPTIBLE]]
                p_k_V = self.node_probs[neighbor_k][self.STATE_MAP[VACCINATED]]
                a_jk = 1.0  # 邻接矩阵元素
                term = c_j * self.beta1 * p_k_S + c_j * self.beta2 * p_k_V
                importance += term * a_jk
            return self.delta * importance
        else:
            # 对于V或R状态节点，传播重要性为0
            return 0.0

    def calculate_intervention_probability(self, node_i):
        """
        计算节点i被干预的概率ρ_i(t) - 根据理论公式(8)
        ρ_i(t) = 1 - Π_{k=1}^{N_r} [1 - P(R_k(t) = i)]
        """
        if self.num_regulators == 0:
            return 0.0

        prob_no_regulator = 1.0
        for regulator_pos in self.regulator_positions:
            if regulator_pos == node_i:
                prob_no_regulator *= 0.0  # 至少有一个监管者在节点i
                break

        return 1.0 - prob_no_regulator

    def initialize_simulation(self):
        """初始化模拟"""
        # 创建网络 - 优化连通性检查，避免无限循环
        self.G = nx.barabasi_albert_graph(self.num_nodes, BA_M)
        # 最多尝试3次创建连通网络
        # 加强网络连通性保证
        max_attempts = 10
        for attempt in range(max_attempts):
            self.G = nx.barabasi_albert_graph(self.num_nodes, BA_M)
            if nx.is_connected(self.G):
                break
        else:
            # 如果还是不通，强制连通
            self.G = nx.connected_watts_strogatz_graph(self.num_nodes, k=8, p=0.1)
            print(f"警告: 使用强制连通网络 (尝试{max_attempts}次后)")
        self.node_degrees = dict(self.G.degree())
        print("正在预计算最短路径...这可能需要一些时间")
        self.all_shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        print(f"最短路径预计算完成，共{len(self.all_shortest_paths)}个节点")
        self.zhijing = max(
            dist for paths in self.all_shortest_paths.values()
            for dist in paths.values()
        )
        # 计算网络特征
        # 初始化节点状态 - 使用微观马尔科夫概率
        self.node_states = {node: SUSCEPTIBLE for node in self.G.nodes()}
        self.node_probs = {}
        for node in self.G.nodes():
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE_MAP[SUSCEPTIBLE]] = 1.0

        # 初始感染节点（选择高度数节点）
        high_degree_nodes = sorted(self.node_degrees.items(), key=lambda x: x[1], reverse=True)[
                            :int(self.num_nodes * 0.05)]
        for node, _ in high_degree_nodes:
            self.node_states[node] = INFECTED
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE_MAP[INFECTED]] = 1.0

        # 初始免疫节点
        remaining_nodes = [n for n in self.G.nodes() if self.node_states[n] == SUSCEPTIBLE]
        if len(remaining_nodes) > 100:
            vaccinated_nodes = random.sample(remaining_nodes, 100)
            for node in vaccinated_nodes:
                self.node_states[node] = VACCINATED
                self.node_probs[node] = np.zeros(4)
                self.node_probs[node][self.STATE_MAP[VACCINATED]] = 1.0

        # 初始化监管者
        if self.num_regulators > 0:
            self.regulator_positions = random.sample(list(self.G.nodes()), self.num_regulators)
            self.regulator_visit_history = [defaultdict(int) for _ in range(self.num_regulators)]
        else:
            self.regulator_positions = []
            self.regulator_visit_history = []

        # 初始化历史记录
        self.last_visit_impact.clear()
        self.last_visit_time.clear()
        for node in self.G.nodes():
            self.last_visit_time[node] = -1000

        # 记录初始状态
        self.record_state()

    def record_state(self):
        """记录当前状态"""
        total = self.num_nodes
        counts = {SUSCEPTIBLE: 0, INFECTED: 0, VACCINATED: 0, RECOVERED: 0}
        for state in self.node_states.values():
            counts[state] += 1

        self.results['S'].append(counts[SUSCEPTIBLE] / total)
        self.results['I'].append(counts[INFECTED] / total)
        self.results['V'].append(counts[VACCINATED] / total)
        self.results['R'].append(counts[RECOVERED] / total)

        # 计算监管效率
        if self.num_regulators > 0:
            current_step = len(self.results['S']) - 1
            efficiency_sum = 0
            for i in range(self.num_regulators):
                current_pos = self.regulator_positions[i]
                importance = self.calculate_node_importance(current_pos, current_step)
                efficiency_sum += importance
            self.results['regulator_efficiency'].append(efficiency_sum / self.num_regulators)
        else:
            self.results['regulator_efficiency'].append(0.0)

    def calculate_avg_distance_to_regulators(self, node):
        """计算节点到所有监管者的平均距离（使用预计算结果）"""
        if self.num_regulators == 0:
            return 0  # 无监管者时返回节点总数作为默认距离

        distances = []
        for reg_pos in self.regulator_positions:
            # 使用预计算的最短路径结果
            if reg_pos in self.all_shortest_paths and node in self.all_shortest_paths[reg_pos]:
                dist = self.all_shortest_paths[reg_pos][node]
                distances.append(max(dist, 1e-6))
            else:
                # 如果预计算中没有找到路径（理论上不应该发生，因为网络是连通的）
                distances.append(self.num_nodes)
        return np.mean(distances) if distances else self.num_nodes

    def calculate_confrontation_factor(self, node):
        """计算对抗因子（指数衰减）"""
        avg_dist = self.calculate_avg_distance_to_regulators(node)
        return  self.xi / (1 + np.exp(-avg_dist))

    def calculate_infection_probabilities(self, node):
        """计算感染概率（微观马尔科夫方法）"""
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return 0.0, 0.0

        c_i = self.calculate_confrontation_factor(node)
        product_q = 1.0
        product_v = 1.0

        for neighbor in neighbors:
            a_ji = 1.0  # 邻接矩阵元素
            p_j_I = self.node_probs[neighbor][self.STATE_MAP[INFECTED]]
            product_q *= (1 - a_ji * c_i * self.beta1 * p_j_I)
            product_v *= (1 - a_ji * c_i * self.beta2 * p_j_I)

        q_i = 1 - product_q  # S→I感染概率
        v_i = 1 - product_v  # V→I突破感染概率
        return max(0.0, min(1.0, q_i)), max(0.0, min(1.0, v_i))

    def calculate_transition_probabilities(self, node):
        """计算节点状态转移概率矩阵 - 考虑监管干预"""
        current_state = self.node_states[node]
        q_i, v_i = self.calculate_infection_probabilities(node)
        rho_i = self.calculate_intervention_probability(node)  # 监管干预概率

        trans_probs = np.zeros(4)

        if current_state == SUSCEPTIBLE:
            # 考虑监管干预：以概率ρ_i转移到V，否则正常传播
            trans_probs[self.STATE_MAP[VACCINATED]] = rho_i
            trans_probs[self.STATE_MAP[SUSCEPTIBLE]] = (1 - rho_i) * (1 - q_i)
            trans_probs[self.STATE_MAP[INFECTED]] = (1 - rho_i) * q_i

        elif current_state == VACCINATED:
            stay_v = 1 - v_i - self.gamma1
            trans_probs[self.STATE_MAP[VACCINATED]] = max(0.0, stay_v)
            trans_probs[self.STATE_MAP[SUSCEPTIBLE]] = self.gamma1
            trans_probs[self.STATE_MAP[INFECTED]] = v_i

        elif current_state == INFECTED:
            # 考虑监管干预：以概率ρ_i转移到R，否则正常恢复
            trans_probs[self.STATE_MAP[RECOVERED]] = rho_i + (1 - rho_i) * self.mu
            trans_probs[self.STATE_MAP[INFECTED]] = (1 - rho_i) * (1 - self.mu)

        elif current_state == RECOVERED:
            trans_probs[self.STATE_MAP[RECOVERED]] = 1 - self.gamma2
            trans_probs[self.STATE_MAP[VACCINATED]] = self.gamma2

        # 归一化概率
        prob_sum = np.sum(trans_probs)
        if prob_sum > 1e-10:
            trans_probs /= prob_sum
        else:
            trans_probs = np.zeros(4)
            trans_probs[self.STATE_MAP[current_state]] = 1.0

        return trans_probs

    def update_mmca_states(self):
        """更新所有节点的状态（MMCA机制）"""
        new_states = {}
        for node in self.G.nodes():
            trans_probs = self.calculate_transition_probabilities(node)
            # 根据概率分布选择新状态
            state_idx = np.random.choice(4, p=trans_probs)
            new_state = [k for k, v in self.STATE_MAP.items() if v == state_idx][0]
            new_states[node] = new_state

            # 更新概率分布
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][state_idx] = 1.0

        self.node_states = new_states

    def regulator_next_move(self, regulator_idx):
        """
        监管者移动决策 - 基于传播重要性的随机游走
        根据理论公式(7): P^r_{i→j}(t) = Ψ_j(t) / Σ_{w∈N(i)} Ψ_w(t)
        """
        current_step = len(self.results['S']) - 1
        current_pos = self.regulator_positions[regulator_idx]

        # 获取当前节点的所有邻居
        neighbors = list(self.G.neighbors(current_pos))
        if not neighbors:
            return current_pos  # 如果没有邻居，停留在原地

        # 计算每个邻居的传播重要性
        importance_scores = {}
        for neighbor in neighbors:
            importance_scores[neighbor] = self.calculate_propagation_importance(neighbor, current_step)

        # 计算总重要性（分母）
        total_importance = sum(importance_scores.values())

        if total_importance <= 1e-10:  # 避免除零
            # 如果所有邻居重要性都为0，随机选择一个邻居
            return random.choice(neighbors)

        # 根据理论公式(7)计算转移概率
        transition_probs = {}
        for neighbor, importance in importance_scores.items():
            transition_probs[neighbor] = importance / total_importance

        # 根据概率分布随机选择下一个节点
        neighbors_list = list(transition_probs.keys())
        prob_list = list(transition_probs.values())

        next_node = np.random.choice(neighbors_list, p=prob_list)
        return next_node

    def calculate_node_importance(self, node, current_step):
        """计算节点重要性 - 现在基于传播重要性"""
        return self.calculate_propagation_importance(node, current_step)

    def execute_regulatory_action(self, node):
        """
        执行监管干预 - 根据理论描述：
        1. 如果节点在状态S，转移到状态V
        2. 如果节点在状态I，转移到状态R
        """
        original_state = self.node_states[node]

        if original_state == SUSCEPTIBLE:
            self.node_states[node] = VACCINATED
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE_MAP[VACCINATED]] = 1.0
            return 1.0
        elif original_state == INFECTED:
            self.node_states[node] = RECOVERED
            self.node_probs[node] = np.zeros(4)
            self.node_probs[node][self.STATE_MAP[RECOVERED]] = 1.0
            return 1.0
        else:
            # 对V或R状态节点无效果
            return 0.0

    def update_regulators(self):
        """更新监管者位置并执行监管动作 - 根据理论公式(9)"""
        if self.num_regulators == 0:
            return

        current_step = len(self.results['S']) - 1

        # 首先更新监管者位置
        new_positions = []
        for i in range(self.num_regulators):
            new_pos = self.regulator_next_move(i)
            new_positions.append(new_pos)

        self.regulator_positions = new_positions

        # 然后执行监管动作
        for i, pos in enumerate(self.regulator_positions):
            # 记录访问历史
            self.regulator_visit_history[i][pos] = current_step
            self.last_visit_time[pos] = current_step

            # 执行监管动作
            impact = self.execute_regulatory_action(pos)
            self.last_visit_impact[pos] = impact

    def step(self):
        """执行一步模拟"""
        # 使用微观马尔科夫方法更新节点状态
        self.update_mmca_states()

        # 更新监管者
        self.update_regulators()

        # 记录状态
        self.record_state()

    def run_simulation(self):
        """运行完整模拟"""
        for _ in range(self.max_steps - 1):
            self.step()
        return self.results


def run_single_simulation_worker(config):
    """Run one simulation repeat in a child process."""
    seed = config.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sim = SIVRRumorSimulation(
        num_nodes=config['network_size'],
        num_regulators=config['regulators'],
        beta1=config['infection_rate'],
        beta2=config['infection_rate'] * 0.3,
        mu=0.08,
        gamma1=0.03,
        gamma2=0.06,
        max_steps=config.get('max_steps', 30),
        delta=config.get('delta', 0.75)
    )

    sim.initialize_simulation()
    results = sim.run_simulation()
    infections = results['I']
    return {
        'final_infection': infections[-1],
        'peak_infection': max(infections),
        'outbreak_duration': sum(1 for density in infections if density > 0.1)
    }


def get_worker_count(repeat_count, max_workers=MAX_WORKERS):
    """返回本次重复实验实际使用的进程数。"""
    if repeat_count <= 0:
        return 0
    if max_workers is None:
        max_workers = min(repeat_count, os.cpu_count() or 1)
    return max(1, min(max_workers, repeat_count))


def run_parallel_simulation_repeats(base_config, repeat_count, max_workers=MAX_WORKERS):
    """多进程运行重复实验，并返回每次实验的指标。"""
    worker_count = get_worker_count(repeat_count, max_workers)
    if worker_count == 0:
        return []

    configs = []
    for _ in range(repeat_count):
        config = base_config.copy()
        config['seed'] = random.randrange(0, 2 ** 32 - 1)
        configs.append(config)

    print(f"    使用 {worker_count} 个进程并行运行 {repeat_count} 次重复实验...")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(run_single_simulation_worker, configs))


class RegulatoryEffectivenessStudy:
    """监管有效性深度研究"""

    def __init__(self):
        self.results = {}
        self.metrics_history = []
        self.max_workers = max(1, int(os.environ.get('SIMULATION_WORKERS', os.cpu_count() or 1)))
        # 自动创建数据存储目录
        if not os.path.exists('simulation_data'):
            os.makedirs('simulation_data')

    def run_study(self):
        """运行完整的监管有效性研究"""
        print("=" * 80)
        print("            SIVR谣言传播模型 - 监管有效性深度研究（改进版）")
        print("=" * 80)

        # 研究设计概述
        study_design = self.get_study_design()
        print("\n研究设计:")
        for item in study_design:
            print(f"  • {item}")

        # 运行核心实验
        print("\n开始运行核心实验...")
        core_results = self.run_core_experiment()

        # 深度分析
        print("\n进行深度分析...")
        analysis_results = self.perform_deep_analysis(core_results)

        # 生成综合报告（含绘图）
        print("\n生成研究报告...")
        self.generate_comprehensive_report(core_results, analysis_results)

        print("\n🎯 研究完成！")
        return core_results, analysis_results

    def get_study_design(self):
        """返回研究设计概述"""
        return [
            "研究问题：监管者数量如何影响谣言控制效果？",
            "实验变量：监管者数量(0, 5, 10, 20, 30, 40, 50)",
            "网络规模：500节点和1000节点",
            "传播风险：低(β=0.2)、中(β=0.3)、高(β=0.4)三个水平",
            "评估指标：最终感染密度、峰值感染、爆发持续时间、控制效率、成本效益",
            "重复实验：每个条件重复3次取平均",
            "分析方法：边际效益分析、最优配置识别、敏感性检验",
            "改进特性：微观马尔科夫传播 + 基于传播重要性的智能监管者移动策略"
        ]

    def get_baseline_performance(self, net_size, beta, num_repeats):
        """获取更准确的基线性能"""
        repeat_results = run_parallel_simulation_repeats({
            'network_size': net_size,
            'regulators': 0,
            'infection_rate': beta,
            'max_steps': 30
        }, num_repeats, self.max_workers)

        peak_infections = [result['peak_infection'] for result in repeat_results]
        final_infections = [result['final_infection'] for result in repeat_results]
        durations = [result['outbreak_duration'] for result in repeat_results]

        return {
            'peak_infection': np.mean(peak_infections),
            'final_infection': np.mean(final_infections),
            'outbreak_duration': np.mean(durations),
            'peak_std': np.std(peak_infections),
            'final_std': np.std(final_infections)
        }

    def get_baseline_final_infection(self, net_size, beta, num_repeats):
        """专门获取基线最终感染密度"""
        try:
            repeat_results = run_parallel_simulation_repeats({
                'network_size': net_size,
                'regulators': 0,
                'infection_rate': beta,
                'max_steps': 30
            }, num_repeats, self.max_workers)
            final_infections = [result['final_infection'] for result in repeat_results]

            return np.mean(final_infections) if final_infections else 0.5
        except Exception as e:
            print(f"获取基线最终感染密度时出错: {e}")
            return 0.5  # 返回合理的默认值

    def get_baseline_final_infection_consistent(self, network_size, infection_rate, repeats, max_steps=80):
        """补充：获取一致的基线最终感染密度（原代码缺失该方法）"""
        try:
            repeat_results = run_parallel_simulation_repeats({
                'network_size': network_size,
                'regulators': 0,
                'infection_rate': infection_rate,
                'max_steps': max_steps
            }, repeats, self.max_workers)
            final_infections = [result['final_infection'] for result in repeat_results]
            return np.mean(final_infections) if final_infections else 0.5
        except Exception as e:
            print(f"计算基线最终感染密度时出错: {e}")
            return 0.5

    def run_core_experiment(self):
        """运行核心实验：监管者数量对控制效果的影响"""
        self.metrics_history = []
        # 实验参数配置
        regulator_levels = [0, 10, 20, 30, 45, 60, 80]
        network_sizes = [400, 1000]
        infection_rates = [0.2, 0.4, 0.6]  # beta1值
        num_repeats = 3  # 重复次数

        # 结果存储结构
        results = {
            'scenarios': [],
            'performance_metrics': defaultdict(list),
            'detailed_records': []
        }

        total_combinations = len(network_sizes) * len(infection_rates) * len(regulator_levels)
        current_combination = 0

        for net_size in network_sizes:
            for beta in infection_rates:
                print(f"\n【当前场景】网络规模={net_size}, 传播率β1={beta}")
                # 获取基线性能（无监管）
                baseline_metrics = self.get_baseline_performance(net_size, beta, num_repeats)
                print(f"  无监管基准值：峰值感染={baseline_metrics['peak_infection']:.3f}")

                for reg_count in regulator_levels:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    print(
                        f"  进度: {progress:.1f}% | 监管者数量={reg_count} | 剩余场景={total_combinations - current_combination}")

                    # 运行监管场景
                    scenario_metrics = self.run_regulatory_scenario(
                        net_size, reg_count, beta, num_repeats, baseline_metrics['peak_infection']
                    )

                    # 记录详细结果
                    scenario_id = f"Net{net_size}_Beta{beta}_Reg{reg_count}"
                    results['scenarios'].append(scenario_id)

                    for metric, value in scenario_metrics.items():
                        results['performance_metrics'][metric].append(value)

                    # 保存详细记录
                    record = {
                        'scenario_id': scenario_id,
                        'network_size': net_size,
                        'infection_rate': beta,
                        'regulators': reg_count,
                        **scenario_metrics
                    }
                    results['detailed_records'].append(record)

                    self.metrics_history.append(record)

        print("\n核心实验完成!")
        return results

    def run_regulatory_scenario(self, network_size, regulators, infection_rate, repeats, baseline_peak):
        """运行单个监管场景"""
        try:
            max_steps = 30
            repeat_results = run_parallel_simulation_repeats({
                'network_size': network_size,
                'regulators': regulators,
                'infection_rate': infection_rate,
                'max_steps': max_steps,
                'delta': 0.75
            }, repeats, self.max_workers)

            final_infections = [result['final_infection'] for result in repeat_results]
            peak_infections = [result['peak_infection'] for result in repeat_results]
            durations = [result['outbreak_duration'] for result in repeat_results]

            # 计算综合指标
            avg_final = np.mean(final_infections)
            avg_peak = np.mean(peak_infections)
            avg_duration = np.mean(durations)

            # 修复：当监管者数量为0时，控制效率应该为0
            if regulators == 0:
                control_efficiency = 0.0
            else:
                baseline_final = self.get_baseline_final_infection_consistent(
                    network_size, infection_rate, repeats, max_steps
                )

                # 允许负效率！
                if baseline_final > 1e-6:
                    control_efficiency = (baseline_final - avg_final) / baseline_final
                    # 不再使用 max(0.0, ...)，允许负值
                else:
                    control_efficiency = 0.0  # 基线为0时的特殊处理

                # 修复成本效益 - 也允许负值
            if regulators == 0:
                cost_effectiveness = 0.0
            else:
                cost_effectiveness = control_efficiency / max(regulators, 1)

            return {
                'final_infection': avg_final,
                'peak_infection': avg_peak,
                'outbreak_duration': avg_duration,
                'control_efficiency': control_efficiency,  # 现在可能是负值
                'cost_effectiveness': cost_effectiveness,  # 现在可能是负值
                'final_std': np.std(final_infections),
                'peak_std': np.std(peak_infections)
            }
        except Exception as e:
            print(f"运行监管场景时出错 (网络规模={network_size}, 监管者={regulators}, 传播率={infection_rate}): {e}")
            # 异常时返回默认值，确保程序继续运行
            return {
                'final_infection': 0.5,
                'peak_infection': 0.5,
                'outbreak_duration': 40,
                'control_efficiency': -0.1,
                'cost_effectiveness': -0.01,
                'final_std': 0.1,
                'peak_std': 0.1
            }

    def perform_deep_analysis(self, core_results):
        """进行深度分析"""
        print("进行深度分析...")

        analysis = {}

        # 1. 边际效益分析
        analysis['marginal_analysis'] = self.analyze_marginal_benefits(core_results)

        # 2. 最优配置识别
        analysis['optimal_configs'] = self.identify_optimal_configurations(core_results)

        # 3. 敏感性分析
        analysis['sensitivity'] = self.analyze_sensitivity(core_results)

        # 4. 统计检验
        analysis['statistical_tests'] = self.perform_statistical_tests(core_results)

        return analysis

    def analyze_marginal_benefits(self, results):
        """分析监管的边际效益"""
        records = results['detailed_records']
        marginal_data = []

        # 按网络规模和感染率分组分析
        for net_size in [400, 1000]:
            for beta in [0.2, 0.4, 0.6]:
                group_data = [r for r in records
                              if r['network_size'] == net_size and
                              r['infection_rate'] == beta]

                group_data.sort(key=lambda x: x['regulators'])

                # 计算边际效益
                for i in range(1, len(group_data)):
                    prev = group_data[i - 1]
                    curr = group_data[i]
                    reg_increase = curr['regulators'] - prev['regulators']

                    if reg_increase > 0:
                        marginal_control = (curr['control_efficiency'] - prev['control_efficiency']) / reg_increase
                        marginal_infection_reduction = (prev['final_infection'] - curr[
                            'final_infection']) / reg_increase

                        marginal_data.append({
                            'network_size': net_size,
                            'infection_rate': beta,
                            'regulator_range': f"{prev['regulators']}-{curr['regulators']}",
                            'marginal_control_gain': marginal_control,
                            'marginal_infection_reduction': marginal_infection_reduction,
                            'current_control_level': curr['control_efficiency']
                        })

        return marginal_data

    def identify_optimal_configurations(self, results):
        records = results['detailed_records']
        optimal_configs = {}

        for net_size in [400, 1000]:
            for beta in [0.2, 0.4, 0.6]:
                # 修正：用浮点数beta匹配，而非字符串
                scenario_data = [r for r in records
                                 if r['network_size'] == net_size and
                                 r['infection_rate'] == beta]  # 关键修改

                if scenario_data:
                    # 找到控制效率最高的配置（排除监管者数量为0的情况）
                    valid_control_data = [r for r in scenario_data if r['regulators'] > 0]
                    if valid_control_data:
                        best_control = max(valid_control_data, key=lambda x: x['control_efficiency'])
                    else:
                        best_control = scenario_data[0]

                    # 找到成本效益最高的配置（排除监管者数量为0的情况）
                    valid_cost_data = [r for r in scenario_data if r['regulators'] > 0 and r['cost_effectiveness'] > 0]
                    if valid_cost_data:
                        best_cost_eff = max(valid_cost_data, key=lambda x: x['cost_effectiveness'])
                    else:
                        best_cost_eff = scenario_data[0]

                    optimal_configs[f'Net{net_size}_Beta{beta}'] = {
                        'best_control': best_control,
                        'best_cost_effectiveness': best_cost_eff  # 键名与绘图方法一致
                    }
        return optimal_configs

    def analyze_sensitivity(self, results):
        records = results['detailed_records']
        sensitivity_analysis = {}

        # 网络规模敏感性：仅计算有监管的数据
        size_sensitivity = {}
        for net_size in [400, 1000]:
            size_data = [r for r in records if r['network_size'] == net_size and r['regulators'] > 0]
            if size_data:
                avg_control = np.mean([r['control_efficiency'] for r in size_data])
                size_sensitivity[net_size] = avg_control

        # 感染率敏感性：仅计算有监管的数据
        beta_sensitivity = {}
        for beta in [0.2, 0.4, 0.6]:
            beta_data = [r for r in records if r['infection_rate'] == beta and r['regulators'] > 0]
            if beta_data:
                avg_control = np.mean([r['control_efficiency'] for r in beta_data])
                beta_sensitivity[beta] = avg_control

        sensitivity_analysis['network_size_impact'] = size_sensitivity
        sensitivity_analysis['infection_rate_impact'] = beta_sensitivity
        return sensitivity_analysis

    def perform_statistical_tests(self, results):
        """执行统计检验"""
        records = results['detailed_records']

        # 分组数据：有监管 vs 无监管
        no_reg_data = [r for r in records if r['regulators'] == 0]
        reg_data = [r for r in records if r['regulators'] > 0]

        if not no_reg_data or not reg_data:
            return {}

        # t检验：检验监管是否显著改善效果
        no_reg_infections = [r['final_infection'] for r in no_reg_data]
        reg_infections = [r['final_infection'] for r in reg_data]

        t_stat, p_value = stats.ttest_ind(no_reg_infections, reg_infections)

        return {
            't_test': {
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }

    def generate_comprehensive_report(self, core_results, analysis_results):
        """生成综合研究报告（含绘图）"""
        print("\n生成综合研究报告...")

        # 创建改进的可视化（绘图文字为英文）
        self.create_improved_visualizations(core_results, analysis_results)

        # 生成中文文本报告
        self.generate_text_report(core_results, analysis_results)

        # 保存详细数据
        self.save_detailed_data(core_results, analysis_results)

    def format_small_number(self, num):
        """格式化小数值，避免显示为0"""
        if abs(num) < 0.001 and num != 0:
            return f"{num:.2e}"
        else:
            return f"{num:.4f}"

    # -------------------------- 绘图函数（使用对数坐标） --------------------------
    def plot_main_effect_improved(self, records, ax):
        """主要效果图（最终感染密度 vs 监管者数量）- 线性坐标"""
        df = pd.DataFrame(records)

        # 使用清晰的配色方案
        color_palette = sns.color_palette("husl", 6)
        line_styles = ['-', '--', '-.']

        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta)]

                if not subset.empty:
                    color_idx = i * 3 + j
                    linestyle = line_styles[j]
                    label = f'Net{net_size} β={beta}'

                    # 误差线显示变异性
                    ax.errorbar(subset['regulators'], subset['final_infection'],
                                yerr=subset['final_std'], marker='o', linewidth=2.5,
                                label=label, capsize=5, capthick=1.5,
                                color=color_palette[color_idx], linestyle=linestyle,
                                alpha=0.8, markersize=6)

        # 坐标轴标签、标题改为英文
        ax.set_xlabel('Number of Regulators', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Infection Density', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Regulator Quantity on Final Infection Density', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=0)

    def plot_control_efficiency_improved(self, records, ax):
        """控制效率图 - 改为线性坐标以显示负值"""
        df = pd.DataFrame(records)

        color_palette = sns.color_palette("Set1", 6)
        line_styles = ['-', '--', '-.']

        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta)]

                if not subset.empty:
                    color_idx = i * 3 + j
                    linestyle = line_styles[j]
                    label = f'Net{net_size} β={beta}'

                    # 改为线性坐标，可以显示负值
                    ax.plot(subset['regulators'], subset['control_efficiency'],
                            marker='s', linewidth=3, label=label,
                            color=color_palette[color_idx], linestyle=linestyle,
                            markersize=8, alpha=0.8)

        # 坐标轴标签、标题
        ax.set_xlabel('Number of Regulators', fontsize=12, fontweight='bold')
        ax.set_ylabel('Control Efficiency', fontsize=12, fontweight='bold')
        ax.set_title('Control Efficiency Analysis (Linear Scale - Shows Negative Values)',
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.4)

        # 添加零线，清晰区分正负效果
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=2)
        ax.text(0.02, 0.02, 'Zero Effect Line', transform=ax.transAxes,
                color='red', fontsize=10)

    def plot_cost_effectiveness_improved(self, records, ax):
        """成本效益图 - 改为线性坐标以显示负值"""
        df = pd.DataFrame(records)

        color_palette = sns.color_palette("dark", 6)

        for i, net_size in enumerate([400, 1000]):
            for j, beta in enumerate([0.2, 0.4, 0.6]):
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta) &
                            (df['regulators'] > 0)]  # 只显示有监管者的情况

                if not subset.empty:
                    color_idx = i * 3 + j
                    label = f'Net{net_size} β={beta}'

                    # 改为线性坐标，可以显示负值
                    ax.plot(subset['regulators'], subset['cost_effectiveness'],
                            marker='^', linewidth=2.5, label=label,
                            color=color_palette[color_idx], markersize=8, alpha=0.8)

        # 坐标轴标签、标题
        ax.set_xlabel('Number of Regulators', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cost-Effectiveness', fontsize=11, fontweight='bold')
        ax.set_title('Cost-Effectiveness Analysis (Linear Scale - Shows Negative Values)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

        # 添加零线
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=2)

    def plot_marginal_benefits_improved(self, marginal_data, ax):
        """边际效益分析图 - 改为线性坐标以显示负值"""
        if not marginal_data:
            ax.text(0.5, 0.5, 'No Marginal Benefit Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Marginal Benefits Analysis (Linear Scale)', fontsize=13, fontweight='bold')
            return

        df = pd.DataFrame(marginal_data)

        # 修复：更新颜色字典和标记字典以匹配实际的网络规模和感染率
        colors = {'400': 'steelblue', '1000': 'coral'}  # 修正网络规模
        markers = {'0.2': 'o', '0.4': 's', '0.6': '^'}  # 修正感染率以匹配实际使用的值

        for net_size in [400, 1000]:
            for beta in [0.2, 0.4, 0.6]:
                subset = df[(df['network_size'] == net_size) &
                            (df['infection_rate'] == beta)]

                if not subset.empty:
                    # 提取监管者范围中点作为x坐标
                    x_positions = []
                    y_values = []
                    for _, row in subset.iterrows():
                        reg_range = row['regulator_range']
                        start, end = map(int, reg_range.split('-'))
                        x_pos = (start + end) / 2
                        x_positions.append(x_pos)
                        y_values.append(row['marginal_control_gain'])  # 现在可能是负值

                    label = f'Net{net_size} β={beta}'
                    color = colors[str(net_size)]
                    marker = markers[str(beta)]

                    ax.plot(x_positions, y_values, marker=marker, linewidth=2,
                            markersize=8, label=label, color=color, alpha=0.8)

        # 坐标轴标签、标题
        ax.set_xlabel('Number of Regulators (Midpoint of Range)', fontsize=11)
        ax.set_ylabel('Marginal Control Benefit', fontsize=11)
        ax.set_title('Marginal Benefits Analysis (Linear Scale - Shows Negative Values)',
                     fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 添加零线
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=2)

    def plot_optimal_configurations_improved(self, optimal_configs, ax):
        """最优配置可视化（表格形式）"""
        if not optimal_configs:
            ax.text(0.5, 0.5, 'No Optimal Configuration Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Optimal Regulatory Configurations', fontsize=13, fontweight='bold')
            return

        # 准备表格数据
        table_data = []
        for scenario, configs in optimal_configs.items():
            parts = scenario.split('_')
            net_size = parts[0][3:]
            beta = parts[1][4:]

            best_control = configs['best_control']
            best_cost_eff = configs['best_cost_effectiveness']

            # 使用格式化函数显示小数值
            table_data.append([
                f'Net{net_size}',
                f'β={beta}',
                int(best_control['regulators']),
                self.format_small_number(best_control['control_efficiency']),
                int(best_cost_eff['regulators']),
                self.format_small_number(best_cost_eff['cost_effectiveness'])
            ])

        # 表格列名改为英文
        columns = ['Network', 'Transmission Rate', 'Optimal Ctrl\nRegulator Num', 'Ctrl Efficiency',
                   'Optimal Cost\nRegulator Num', 'Cost-Effectiveness']

        # 隐藏坐标轴
        ax.axis('tight')
        ax.axis('off')

        # 创建表格
        table = ax.table(cellText=table_data,
                         colLabels=columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # 表头样式
        for col in range(len(columns)):
            table[(0, col)].set_facecolor('#4F81BD')
            table[(0, col)].set_text_props(weight='bold', color='white')

        # 行交替颜色
        for row in range(1, len(table_data) + 1):
            color = '#DCE6F1' if row % 2 == 0 else '#EBF1DE'
            for col in range(len(columns)):
                table[(row, col)].set_facecolor(color)

        # 表格标题改为英文
        ax.set_title('Identification of Optimal Regulatory Configurations\n(Control-Optimal vs Cost-Effective-Optimal)',
                     fontsize=13, fontweight='bold', pad=20)

    def plot_sensitivity_analysis_improved(self, sensitivity, ax):
        """敏感性分析图 - 对数坐标"""
        size_data = sensitivity.get('network_size_impact', {})
        beta_data = sensitivity.get('infection_rate_impact', {})

        if not size_data and not beta_data:
            ax.text(0.5, 0.5, 'No Sensitivity Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensitivity Analysis (Log Scale)', fontsize=13, fontweight='bold')
            return

        # 准备柱状图数据
        categories = []
        values = []
        colors = []

        # 网络规模敏感性
        if size_data:
            for size, value in size_data.items():
                categories.append(f'Net{size}')
                values.append(max(value, 1e-10))  # 避免0值
                colors.append('skyblue')

        # 感染率敏感性
        if beta_data:
            for beta, value in beta_data.items():
                categories.append(f'β{beta}')
                values.append(max(value, 1e-10))  # 避免0值
                colors.append('lightcoral')

        x = np.arange(len(categories))

        # 绘制柱状图（对数坐标）
        bars = ax.bar(x, values, alpha=0.7, color=colors, edgecolor='black')
        ax.set_yscale('log')  # 设置y轴为对数坐标

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    self.format_small_number(value), ha='center', va='bottom', fontsize=9)

        # 图例改为英文
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', alpha=0.7, label='Network Size Impact'),
            Patch(facecolor='lightcoral', alpha=0.7, label='Transmission Rate Impact')
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        # 坐标轴标签、标题改为英文
        ax.set_xlabel('Experimental Conditions', fontsize=11)
        ax.set_ylabel('Average Control Efficiency (log scale)', fontsize=11)
        ax.set_title('Sensitivity Analysis of Regulatory Effectiveness (Log Scale)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    def create_improved_visualizations(self, core_results, analysis_results):
        records = core_results['detailed_records']
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            'SIVR Rumor Propagation Model - Comprehensive Analysis of Regulatory Effectiveness (Improved Version)',
            fontsize=18, fontweight='bold', y=0.95)

        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        # 1. 最终感染密度图（线性坐标）
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_main_effect_improved(records, ax1)

        # 2. 控制效率图（对数坐标）
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_control_efficiency_improved(records, ax2)

        # 3. 成本效益图（对数坐标）
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_cost_effectiveness_improved(records, ax3)

        # 4. 边际效益图（对数坐标）
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_marginal_benefits_improved(analysis_results['marginal_analysis'], ax4)

        # 5. 最优配置表格
        ax5 = fig.add_subplot(gs[1, 2:])
        self.plot_optimal_configurations_improved(analysis_results['optimal_configs'], ax5)

        # 6. 敏感性分析图（对数坐标）
        ax6 = fig.add_subplot(gs[2, :])
        self.plot_sensitivity_analysis_improved(analysis_results['sensitivity'], ax6)

        plt.tight_layout()
        plt.savefig('监管有效性深度分析_改进版_基于传播重要性.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_all_experimental_data(self, core_results, analysis_results):
        """保存所有实验数据到单个txt文件"""
        data_file_path = os.path.join('simulation_data', 'all_experimental_data.txt')

        with open(data_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SIVR谣言传播模型 - 所有实验数据汇总\n")
            f.write("=" * 80 + "\n\n")

            # 1. 核心实验数据
            f.write("一、核心实验数据\n")
            f.write("-" * 40 + "\n")
            records = core_results['detailed_records']
            for record in records:
                f.write(f"场景: {record['scenario_id']}\n")
                f.write(f"  网络规模: {record['network_size']}\n")
                f.write(f"  传播率: {record['infection_rate']}\n")
                f.write(f"  监管者数量: {record['regulators']}\n")
                f.write(f"  最终感染密度: {self.format_small_number(record['final_infection'])}\n")
                f.write(f"  峰值感染: {self.format_small_number(record['peak_infection'])}\n")
                f.write(f"  爆发持续时间: {record['outbreak_duration']}\n")
                f.write(f"  控制效率: {self.format_small_number(record['control_efficiency'])}\n")
                f.write(f"  成本效益: {self.format_small_number(record['cost_effectiveness'])}\n")
                f.write(f"  最终感染标准差: {self.format_small_number(record['final_std'])}\n")
                f.write(f"  峰值感染标准差: {self.format_small_number(record['peak_std'])}\n")
                f.write("\n")

            # 2. 边际效益分析数据
            f.write("\n二、边际效益分析数据\n")
            f.write("-" * 40 + "\n")
            marginal_data = analysis_results.get('marginal_analysis', [])
            for data in marginal_data:
                f.write(f"网络规模: {data['network_size']}, 传播率: {data['infection_rate']}\n")
                f.write(f"  监管者范围: {data['regulator_range']}\n")
                f.write(f"  边际控制增益: {self.format_small_number(data['marginal_control_gain'])}\n")
                f.write(f"  边际感染减少: {self.format_small_number(data['marginal_infection_reduction'])}\n")
                f.write(f"  当前控制水平: {self.format_small_number(data['current_control_level'])}\n")
                f.write("\n")

            # 3. 最优配置数据
            f.write("\n三、最优配置数据\n")
            f.write("-" * 40 + "\n")
            optimal_configs = analysis_results.get('optimal_configs', {})
            for scenario, configs in optimal_configs.items():
                f.write(f"场景: {scenario}\n")
                best_control = configs['best_control']
                best_cost_eff = configs['best_cost_effectiveness']
                f.write(f"  最优控制配置:\n")
                f.write(f"    监管者数量: {best_control['regulators']}\n")
                f.write(f"    控制效率: {self.format_small_number(best_control['control_efficiency'])}\n")
                f.write(f"  最优成本效益配置:\n")
                f.write(f"    监管者数量: {best_cost_eff['regulators']}\n")
                f.write(f"    成本效益: {self.format_small_number(best_cost_eff['cost_effectiveness'])}\n")
                f.write("\n")

            # 4. 敏感性分析数据
            f.write("\n四、敏感性分析数据\n")
            f.write("-" * 40 + "\n")
            sensitivity = analysis_results.get('sensitivity', {})
            size_impact = sensitivity.get('network_size_impact', {})
            beta_impact = sensitivity.get('infection_rate_impact', {})

            f.write("网络规模影响:\n")
            for size, value in size_impact.items():
                f.write(f"  网络规模 {size}: {self.format_small_number(value)}\n")

            f.write("传播率影响:\n")
            for beta, value in beta_impact.items():
                f.write(f"  传播率 {beta}: {self.format_small_number(value)}\n")

            # 5. 统计检验结果
            f.write("\n五、统计检验结果\n")
            f.write("-" * 40 + "\n")
            statistical_tests = analysis_results.get('statistical_tests', {})
            t_test = statistical_tests.get('t_test', {})
            if t_test:
                f.write(f"t检验统计量: {t_test.get('statistic', 'N/A')}\n")
                f.write(f"p值: {t_test.get('p_value', 'N/A')}\n")
                f.write(f"是否显著: {t_test.get('significant', 'N/A')}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("数据保存完成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")

        print(f"所有实验数据已保存到: {data_file_path}")

    # -------------------------- 报告与数据保存（保持中文） --------------------------
    def generate_text_report(self, core_results, analysis_results):
        """生成中文文本报告"""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("           SIVR谣言传播模型 - 监管有效性研究报告")
        report_lines.append("=" * 80)

        # 执行摘要
        report_lines.append("\n一、执行摘要")
        report_lines.append("本研究系统分析了监管者数量对谣言传播控制效果的影响。")
        report_lines.append("通过多场景、多指标的实验设计，揭示了监管的有效性、经济性和最优配置。")
        report_lines.append("【新特性】监管者移动策略基于传播重要性Ψ_j(t)的随机游走。")

        # 主要发现
        report_lines.append("\n二、主要发现")
        marginal_data = analysis_results.get('marginal_analysis', [])
        if marginal_data:
            avg_marginal = np.mean([d['marginal_control_gain'] for d in marginal_data])
            report_lines.append(f"1. 平均边际控制效益: {self.format_small_number(avg_marginal)}")
        optimal_configs = analysis_results.get('optimal_configs', {})
        if optimal_configs:
            report_lines.append("2. 最优监管配置识别完成")
        statistical_tests = analysis_results.get('statistical_tests', {})
        t_test = statistical_tests.get('t_test', {})
        if t_test.get('significant', False):
            report_lines.append("3. 统计检验证实监管效果显著 (p < 0.05)")
        else:
            report_lines.append("3. 统计检验显示监管效果不显著")

        # 详细数据摘要
        report_lines.append("\n三、详细数据摘要")
        records = core_results['detailed_records']
        df = pd.DataFrame(records)
        if not df.empty:
            summary_stats = df.groupby('regulators').agg({
                'final_infection': ['mean', 'std'],
                'control_efficiency': 'mean',
                'cost_effectiveness': 'mean'
            }).round(6)  # 增加小数位数避免显示为0
            report_lines.append("\n监管效果统计摘要:")
            report_lines.append(str(summary_stats))

        # 保存报告到指定目录
        report_path = os.path.join('simulation_data', '监管有效性研究报告_基于传播重要性.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"文本报告已保存为: {report_path}")

    def save_detailed_data(self, core_results, analysis_results):
        """保存详细数据到CSV文件和所有实验数据到txt"""
        # 核心实验数据
        df_records = pd.DataFrame(core_results['detailed_records'])
        records_path = os.path.join('simulation_data', '核心实验数据_基于传播重要性.csv')
        df_records.to_csv(records_path, index=False, encoding='utf-8-sig')

        # 边际效益数据
        if analysis_results.get('marginal_analysis'):
            df_marginal = pd.DataFrame(analysis_results['marginal_analysis'])
            marginal_path = os.path.join('simulation_data', '边际效益分析数据_基于传播重要性.csv')
            df_marginal.to_csv(marginal_path, index=False, encoding='utf-8-sig')

        # 保存所有实验数据到单个txt文件
        self.save_all_experimental_data(core_results, analysis_results)

        print("详细数据已保存到:")
        print(f"  - {records_path}")
        if analysis_results.get('marginal_analysis'):
            print(f"  - {marginal_path}")
        print(f"  - simulation_data/all_experimental_data.txt (所有实验数据汇总)")


if __name__ == "__main__":
    freeze_support()
    study = RegulatoryEffectivenessStudy()
    core_results, analysis_results = study.run_study()

    print("\n" + "=" * 80)
    print("研究输出文件:")
    print("  - 监管有效性深度分析_改进版_基于传播重要性.png (主要可视化结果)")
    print("  - simulation_data/监管有效性研究报告_基于传播重要性.txt (文本报告)")
    print("  - simulation_data/核心实验数据_基于传播重要性.csv (原始实验数据)")
    print("  - simulation_data/边际效益分析数据_基于传播重要性.csv (边际效益分析数据)")
    print("  - simulation_data/all_experimental_data.txt (所有实验数据汇总)")
    print("=" * 80)
