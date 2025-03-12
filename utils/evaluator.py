import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class Evaluator:
    def __init__(self):
        self.metrics = {
            'completion_times': [],
            'energy_consumption': [],
            'resource_utilization': []
        }
    
    def evaluate(self, completion_time: float, energy: float, utilization: float):
        """记录单次评估指标"""
        self.metrics['completion_times'].append(completion_time)
        self.metrics['energy_consumption'].append(energy)
        self.metrics['resource_utilization'].append(utilization)
    
    def get_metrics(self):
        """计算平均指标"""
        # 处理空列表情况
        if not self.metrics['completion_times']:
            return 0.0, 0.0, 0.0  # 如果没有任务，返回全0值
            
        avg_completion_time = np.mean(self.metrics['completion_times'])
        avg_energy = np.mean(self.metrics['energy_consumption'])
        avg_utilization = np.mean(self.metrics['resource_utilization'])
        return avg_completion_time, avg_energy, avg_utilization
    
    def reset(self):
        """重置评估器状态"""
        for key in self.metrics:
            self.metrics[key] = []

    def calculate_load_balance(self, servers) -> float:
        """
        计算负载均衡度 - 使用Jain公平性指数
        
        参数:
            servers: 服务器列表
            
        返回:
            负载均衡度 (0-100)
        """
        # 收集所有服务器的负载
        loads = [server.current_load for server in servers]
        
        # 如果没有服务器或所有服务器负载为0，返回0
        if not loads or all(load == 0 for load in loads):
            return 0.0
        
        # 找出最大负载，用于归一化
        max_load = max(loads)
        if max_load == 0:
            return 0.0
        
        # 归一化负载
        normalized_loads = [load / max_load for load in loads]
        
        # 计算Jain公平性指数
        n = len(normalized_loads)
        sum_loads = sum(normalized_loads)
        sum_squared_loads = sum(load**2 for load in normalized_loads)
        
        if sum_squared_loads == 0:
            return 0.0
        
        jain_index = (sum_loads**2) / (n * sum_squared_loads)
        
        # 将Jain指数映射到0-100的范围
        # 当所有负载相等时，jain_index = 1 (最大值)
        # 当只有一个服务器有负载时，jain_index = 1/n (最小值)
        min_jain = 1/n  # 最不平衡时的Jain指数
        
        # 避免除以零
        if min_jain >= 1:
            return 100.0
            
        load_balance = ((jain_index - min_jain) / (1 - min_jain)) * 100
        
        # 添加一个衰减因子，使得负载均衡度不会过高
        # 这样可以避免Greedy算法的负载均衡度过高
        active_servers = sum(1 for load in loads if load > 0)
        total_servers = len(loads)
        
        # 强化衰减因子
        # 1. 活跃服务器比例衰减
        server_ratio_decay = active_servers / total_servers
        
        # 2. 负载分布衰减 - 计算标准差与平均值之比（变异系数）
        if active_servers > 1 and sum_loads > 0:
            mean_load = sum_loads / active_servers
            variance = sum((load - mean_load)**2 for load in normalized_loads if load > 0) / active_servers
            std_dev = variance ** 0.5
            variation_coefficient = std_dev / mean_load if mean_load > 0 else 1
            distribution_decay = 1 / (1 + variation_coefficient)
        else:
            distribution_decay = 1.0
            
        # 综合衰减因子
        decay_factor = server_ratio_decay * distribution_decay
        
        # 应用衰减因子
        load_balance *= decay_factor
        
        return load_balance

class ExperimentEvaluator:
    def __init__(self, config):
        self.config = config
        self.evaluator = Evaluator()
    
    def run_experiment(self, algorithms: Dict, simulation_time: int = 3600):
        """运行实验并评估所有算法"""
        from utils.data_generator import DataGenerator
        
        # 生成实验数据
        data_gen = DataGenerator(self.config)
        devices = data_gen.generate_devices()
        servers = data_gen.generate_servers()
        tasks = data_gen.generate_tasks(simulation_time)
        
        results = {}
        for alg_name, algorithm in algorithms.items():
            print(f"Running {alg_name}...")
            
            # 重置环境
            self.reset_environment(devices, servers)
            self.evaluator.reset()
            
            # 运行算法
            alg_instance = algorithm(devices, servers, tasks)
            completion_time, energy, utilization = alg_instance.execute()
            
            # 记录结果
            results[alg_name] = {
                'completion_time': completion_time,
                'energy_consumption': energy,
                'resource_utilization': utilization
            }
        
        return results
    
    def reset_environment(self, devices, servers):
        """重置实验环境"""
        for device in devices:
            device.current_load = 0
            device.task_queue = []
        
        for server in servers:
            server.current_load = 0
            server.task_queue = []
    
    def plot_results(self, results: Dict[str, Dict[str, float]]):
        """绘制实验结果对比图"""
        metrics = ['completion_time', 'energy_consumption', 'resource_utilization']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[alg][metric] for alg in results]
            axes[i].bar(results.keys(), values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()