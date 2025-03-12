from typing import List, Tuple
import numpy as np
from utils.evaluator import Evaluator
import random

class GreedyAlgorithm:
    def __init__(self, devices: List, servers: List, tasks: List, config=None):
        """
        贪心算法实现
        
        参数:
            devices: 终端设备列表
            servers: 边缘服务器列表
            tasks: 任务列表
            config: 实验配置，可选
        """
        self.devices = devices
        self.servers = servers
        self.tasks = tasks
        self.config = config
        
        # 使用统一的评估器
        self.evaluator = Evaluator()
        
        self.server_loads = {server.id: 0 for server in servers}
        
    def execute(self) -> Tuple[float, float, float]:
        """执行贪心算法，基于任务优先级和服务器可用资源进行卸载决策"""
        # 按优先级对任务排序（优先级高的先处理）
        sorted_tasks = sorted(self.tasks, key=lambda x: x.priority, reverse=True)
        
        total_delay = 0
        total_energy = 0
        total_tasks = len(sorted_tasks)
        
        # 初始化服务器负载
        for server in self.servers:
            server.current_load = 0
            self.server_loads[server.id] = 0
        
        # 处理所有任务
        for task in sorted_tasks:
            # 找到任务所属设备
            device = self.find_device(task)
            if not device:
                continue
                
            # 根据贪心策略决定是否卸载
            # 1. 查找可用的服务器
            available_servers = [s for s in self.servers if s.current_load / s.cpu_freq < 0.95]  # 提高阈值到95%
            
            if not available_servers:
                # 如果没有可用服务器，在本地执行
                delay, energy = self.execute_locally(device, task)
            else:
                # 修改服务器选择策略：不仅考虑负载，还考虑性能和距离
                # 计算每个服务器的综合得分 (性能得分 - 负载惩罚)
                server_scores = []
                for server in available_servers:
                    # 性能得分 - 基于CPU频率
                    performance_score = server.cpu_freq / 1e10  # 归一化到0-1范围
                    
                    # 负载惩罚 - 基于当前负载率
                    load_penalty = (server.current_load / server.cpu_freq) * 0.5  # 降低负载均衡的权重
                    
                    # 综合得分
                    score = performance_score - load_penalty
                    server_scores.append((server, score))
                
                # 选择得分最高的服务器
                best_server = max(server_scores, key=lambda x: x[1])[0]
                
                # 计算本地执行和卸载执行的成本
                local_cost = self.calculate_local_cost(device, task)
                offload_cost = self.calculate_offload_cost(device, best_server, task)
                
                # 选择成本更低的方式执行
                if local_cost <= offload_cost:
                    # 本地执行
                    delay, energy = self.execute_locally(device, task)
                else:
                    # 卸载执行
                    delay, energy = self.execute_offload(device, best_server, task)
            
            total_delay += delay
            total_energy += energy
        
        # 计算资源利用率
        resource_utilization = self.calculate_resource_utilization()
        
        # 计算平均延迟
        avg_delay = total_delay / total_tasks if total_tasks > 0 else 0
        
        # 打印最终结果
        print(f"Greedy - 平均完成时延: {avg_delay:.4f}秒, 总能耗: {total_energy:.4e}, 负载均衡度: {resource_utilization:.2f}%")
        
        return avg_delay, total_energy, resource_utilization
    
    def calculate_resource_utilization(self) -> float:
        """计算负载均衡度 - 使用统一的评估器"""
        balance = self.evaluator.calculate_load_balance(self.servers)
        
        # 打印服务器负载信息
        server_loads = [server.current_load for server in self.servers]
        print(f"Greedy - 服务器负载: {server_loads}, 均衡度: {balance:.2f}")
        
        return balance
    
    def calculate_local_cost(self, device, task) -> float:
        """计算本地执行成本（时间和能耗的加权和）"""
        execution_time = task.computing_cycles / device.cpu_freq
        energy = device.energy_coefficient * (device.cpu_freq ** 2) * execution_time
        
        # 成本是时间和归一化能耗的加权和
        return execution_time + energy * 0.5
    
    def calculate_offload_cost(self, device, server, task) -> float:
        """计算卸载执行成本（时间和能耗的加权和）"""
        transmission_time = task.data_size / server.bandwidth
        execution_time = task.computing_cycles / server.cpu_freq
        transmission_energy = device.transmit_power * transmission_time
        
        # 成本是总时间和归一化能耗的加权和
        return (transmission_time + execution_time) + transmission_energy * 0.5
    
    def execute_locally(self, device, task) -> Tuple[float, float]:
        """本地执行任务"""
        # 计算本地执行时间
        execution_time = task.computing_cycles / device.cpu_freq
        
        # 计算能耗
        computing_energy = device.energy_coefficient * (device.cpu_freq ** 2) * execution_time
        
        # 根据任务计算量调整能耗
        task_complexity_factor = task.computing_cycles / 1e6
        
        # 计算总能耗 - 与所有算法使用统一的能耗计算方法
        total_energy = computing_energy * task_complexity_factor * 1e9
        
        # 更新设备负载
        device.current_load += task.computing_cycles
        
        return execution_time, total_energy
    
    def execute_offload(self, device, server, task) -> Tuple[float, float]:
        """卸载执行任务"""
        # 计算传输时间和执行时间
        transmission_time = task.data_size / server.bandwidth
        
        # 增加网络延迟影响
        network_latency = random.uniform(0.01, 0.03)  # 10-30毫秒的网络延迟
        transmission_time += network_latency
        
        execution_time = task.computing_cycles / server.cpu_freq
        
        # 计算总时间（传输+执行）
        total_time = transmission_time + execution_time
        
        # 计算传输能耗
        transmission_energy = device.transmit_power * transmission_time
        
        # 计算服务器执行能耗
        execution_energy = server.energy_coefficient * task.computing_cycles if hasattr(server, 'energy_coefficient') else 0
        
        # 计算总能耗 - 使用线性计算方式，与GNN-DRL保持一致
        total_energy = (transmission_energy + execution_energy) * 1e3
        
        # 更新服务器负载
        server.current_load += task.computing_cycles
        self.server_loads[server.id] += task.computing_cycles
        
        # 打印服务器负载更新信息
        print(f"Greedy - 服务器 {server.id} 负载已更新: {server.current_load:.2e}/{server.cpu_freq:.2e} (负载比例={(server.current_load/server.cpu_freq*100):.2f}%)")
        
        return total_time, total_energy
    
    def find_device(self, task):
        """查找任务所属的设备"""
        for device in self.devices:
            if task in device.task_queue:
                return device
        return None