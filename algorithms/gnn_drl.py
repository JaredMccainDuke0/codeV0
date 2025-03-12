import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F
import random

from models.gnn import TaskGNN
from models.drl import DRLAgent
from utils.evaluator import Evaluator

class GNNDRL:
    def __init__(self, devices: List, servers: List, tasks: List, config=None):
        """
        基于GNN和深度强化学习的任务卸载算法
        
        参数:
            devices: 终端设备列表
            servers: 边缘服务器列表
            tasks: 任务列表
            config: 实验配置
        """
        self.devices = devices
        self.servers = servers
        self.tasks = tasks
        self.config = config
        self.evaluator = Evaluator()
        
        # 初始化服务器负载字典
        self.server_loads = {i: 0 for i in range(len(servers))}
        
        # 添加调试模式标志，默认为False
        self.debug_mode = False
        
        # 初始化GNN模型
        self.input_dim = 5  # 任务特征维度：计算量、数据大小、截止时间、优先级、依赖数
        self.hidden_dim = 32  # 隐藏层维度
        self.n_heads = 4  # 注意力头数
        
        # 设置边缘服务器的数量
        self.num_servers = len(servers)
        
        # 计算状态向量维度，增加服务器负载信息
        # 原始状态: 任务嵌入(hidden_dim) + 设备特征(4) = 36
        # 新增维度: 服务器负载特征(num_servers)
        self.state_dim = self.hidden_dim + 4 + self.num_servers
        
        print(f"DRL代理状态维度: {self.state_dim}")
        
        # 初始化GNN模型
        self.gnn = TaskGNN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, n_heads=self.n_heads)
        
        # 初始化DRL代理
        # 动作空间: 本地执行(1) + 卸载到边缘服务器(num_servers)
        action_dim = 1 + len(servers)
        self.drl_agent = DRLAgent(self.state_dim, action_dim)
        
        # 初始化所有服务器负载为0
        for server in self.servers:
            server.current_load = 0
        
        # 加载预训练模型
        self._load_pretrained_models()
        
    def _build_task_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建任务之间的关系图，用于GNN处理"""
        # 如果没有任务，返回空张量
        if len(self.tasks) == 0:
            print("任务列表为空，无法构建任务图")
            return torch.zeros((0, self.input_dim)), torch.zeros((2, 0), dtype=torch.long)
        
        # 记录任务数量，用于调试
        num_tasks = len(self.tasks)
        print(f"构建任务图 - 任务数量: {num_tasks}")
        
        # 检查任务是否有有效的依赖关系
        has_dependencies = False
        for task in self.tasks:
            if hasattr(task, 'dependencies') and len(task.dependencies) > 0:
                has_dependencies = True
                break
                
        if not has_dependencies:
            print("警告: 所有任务都没有依赖关系，任务图将是无边的")
            
        # 提取任务特征
        task_features = []
        for i, task in enumerate(self.tasks):
            # 确保任务特征维度与input_dim一致
            features = [
                task.computing_cycles / 1e9,  # 归一化计算量
                task.input_data_size / 1e6,   # 归一化数据大小
                task.deadline / 10 if hasattr(task, 'deadline') else 0.5,  # 归一化截止时间
                task.priority / 10 if hasattr(task, 'priority') else 0.5,  # 归一化优先级
                len(task.dependencies) / 10 if hasattr(task, 'dependencies') else 0  # 归一化依赖数量
            ]
            task_features.append(features)
            
            # 打印任务特征，用于调试
            if self.debug_mode:
                print(f"任务 {i} (ID: {task.id}) 特征: {features}")
                if hasattr(task, 'dependencies'):
                    print(f"  依赖任务: {task.dependencies}")
        
        # 转换为张量
        node_features = torch.tensor(task_features, dtype=torch.float)
        
        # 构建任务之间的依赖关系（边）
        edge_index = []
        for i, task in enumerate(self.tasks):
            if hasattr(task, 'dependencies'):
                for dep_id in task.dependencies:
                    found = False
                    for j, dep_task in enumerate(self.tasks):
                        if dep_task.id == dep_id:
                            # 从依赖任务到当前任务的边
                            edge_index.append([j, i])
                            found = True
                            break
                    if not found and self.debug_mode:
                        print(f"警告: 任务 {task.id} 依赖的任务 {dep_id} 不在当前任务列表中")
        
        # 打印边的信息，用于调试
        if self.debug_mode:
            print(f"边数量: {len(edge_index)}")
            for edge in edge_index:
                print(f"  边: {edge[0]} -> {edge[1]}")
        
        # 如果没有边，返回空边索引
        if len(edge_index) == 0:
            print("任务图中没有边，这可能会影响GNN的性能")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index).t()
        
        return node_features, edge_index

    def execute(self) -> Tuple[float, float, float]:
        """执行GNN-DRL任务卸载算法"""
        # 检查任务列表是否为空
        if len(self.tasks) == 0:
            print(f"GNN-DRL - 任务数量为0，返回默认值")
            return 0.0, 0.0, 0.0  # 返回默认值：平均延迟、能耗、负载均衡度均为0
        
        # 打印任务详情，用于调试
        print(f"GNN-DRL - 处理任务数量: {len(self.tasks)}")
        for i, task in enumerate(self.tasks):
            print(f"  任务 {i}: ID={task.id}, 计算量={task.computing_cycles:.2e}, 数据大小={task.input_data_size:.2e}")
            if hasattr(task, 'dependencies'):
                print(f"    依赖任务: {task.dependencies}")
        
        # 启用调试信息
        self.debug_mode = True
        
        # 设置DRL代理为测试模式，但保留一定探索率以促进负载均衡
        if hasattr(self, 'drl_agent') and self.drl_agent is not None:
            self.drl_agent.set_test_mode(True)
        
        # 初始化服务器负载
        for server in self.servers:
            server.current_load = 0
        
        # 按照拓扑顺序对任务进行排序
        task_indices = self._topological_sort()
        print(f"拓扑排序后的任务顺序: {task_indices}")
        
        # 构建任务图
        node_features, edge_index = self._build_task_graph()
        
        # 检查任务图是否有效
        if node_features.numel() == 0:
            print(f"GNN-DRL - 任务图节点为空，返回默认值")
            return 0.0, 0.0, 0.0
            
        if edge_index.numel() == 0:
            print(f"GNN-DRL - 任务图边为空，使用无边的任务图")
        
        # 打印任务图信息
        print(f"任务图 - 节点数: {node_features.size(0)}, 边数: {edge_index.size(1) if edge_index.dim() > 0 else 0}")
        
        # 如果没有边，我们仍然可以处理孤立的节点(使用自循环)
        if edge_index.size(1) == 0 and node_features.size(0) > 0:
            print("为孤立节点添加自循环边")
            # 为每个节点添加自循环
            self_loops = [[i, i] for i in range(node_features.size(0))]
            edge_index = torch.tensor(self_loops, dtype=torch.long).t()
            print(f"添加自循环后的边数: {edge_index.size(1)}")
        
        # 使用GNN生成任务嵌入表示
        with torch.no_grad():
            try:
                task_embeddings = self.gnn(node_features, edge_index)
                
                # 检查嵌入结果
                if task_embeddings.numel() == 0:
                    print(f"GNN-DRL - 嵌入生成失败，返回默认值")
                    return 0.0, 0.0, 0.0
            except Exception as e:
                print(f"GNN-DRL - 生成任务嵌入时出错: {e}")
                return 0.0, 0.0, 0.0
        
        total_delay = 0
        total_energy = 0
        total_tasks = len(self.tasks)
        processed_tasks = 0  # 添加已处理任务计数
        
        # 按拓扑顺序处理任务
        for idx in task_indices:
            task = self.tasks[idx]
            
            # 找到任务所属设备
            device = self.find_device(task)
            if not device:
                continue
            
            processed_tasks += 1  # 增加已处理任务计数
            
            # 获取当前任务的嵌入表示
            task_embedding = task_embeddings[idx]
            
            # 获取当前状态
            state = self._get_current_state(task_embedding, device)
            
            # 检查DRL代理是否可用
            if not hasattr(self, 'drl_agent') or self.drl_agent is None:
                print("警告: DRL代理不可用，使用随机策略")
                action = random.randint(0, len(self.servers))  # 随机选择本地执行或某个服务器
            else:
                # 选择动作（本地执行或卸载到特定服务器）
                action = self._select_action(state)
            
            # 根据所选动作执行任务
            if action == 0:
                # 本地执行
                delay, energy = self._execute_locally(device, task)
            else:
                # 卸载到服务器执行
                server_idx = action - 1
                if server_idx < len(self.servers):
                    server = self.servers[server_idx]
                    delay, energy = self._execute_offload(device, server, task)
                else:
                    # 如果服务器索引无效，默认本地执行
                    delay, energy = self._execute_locally(device, task)
            
            total_delay += delay
            total_energy += energy
            
            # 打印当前服务器负载分布
            if self.debug_mode and processed_tasks % 5 == 0:  # 每处理5个任务打印一次负载情况
                print(f"GNN-DRL - 任务进度: {processed_tasks}/{total_tasks}")
                for i, server in enumerate(self.servers):
                    load_percentage = server.current_load / server.cpu_freq * 100
                    print(f"  服务器 {i} 负载: {load_percentage:.2f}%")
        
        # 如果没有任务被处理，返回默认值
        if processed_tasks == 0:
            print("GNN-DRL - 警告: 没有任务被处理")
            return 0.0, 0.0, 0.0  # 返回默认值：平均延迟、能耗、负载均衡度均为0
        
        # 计算负载均衡度
        load_variance = self._calculate_load_variance()
        resource_utilization = load_variance  # 直接使用计算结果
        
        # 计算平均延迟
        avg_delay = total_delay / total_tasks if total_tasks > 0 else 0
        
        # 打印最终结果
        print(f"GNN-DRL - 平均完成时延: {avg_delay:.4f}秒, 总能耗: {total_energy:.4e}, 负载均衡度: {resource_utilization:.2f}%")
        
        # 打印最终服务器负载分布
        print("GNN-DRL - 最终服务器负载分布:")
        for i, server in enumerate(self.servers):
            load_percentage = server.current_load / server.cpu_freq * 100
            print(f"  服务器 {i} 负载: {load_percentage:.2f}%")
        
        return avg_delay, total_energy, resource_utilization
    
    def _topological_sort(self) -> List[int]:
        """对任务进行拓扑排序"""
        # 构建依赖图
        graph = {i: [] for i in range(len(self.tasks))}
        in_degree = {i: 0 for i in range(len(self.tasks))}
        
        # 填充图和入度信息
        for i, task in enumerate(self.tasks):
            for dep_id in task.dependencies:
                for j, dep_task in enumerate(self.tasks):
                    if dep_task.id == dep_id:
                        graph[j].append(i)
                        in_degree[i] += 1
        
        # 拓扑排序
        queue = [i for i, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            curr = queue.pop(0)
            result.append(curr)
            
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 如果存在环，使用任务ID作为排序标准
        if len(result) < len(self.tasks):
            remaining = [i for i in range(len(self.tasks)) if i not in result]
            remaining.sort(key=lambda i: self.tasks[i].id)
            result.extend(remaining)
        
        return result
    
    def _get_current_state(self, task_embedding: torch.Tensor, device) -> torch.Tensor:
        """生成当前状态表示，增强对服务器负载的感知"""
        # 任务嵌入特征
        task_features = task_embedding.float()
        
        # 调试输出
        if self.debug_mode:
            print(f"Debug - 任务嵌入形状: {task_features.shape}")
        
        # 确保任务嵌入维度正确
        # GATConv输出可能是hidden_dim或hidden_dim * n_heads维度
        expected_dim = self.hidden_dim  # 期望的嵌入维度
        
        if task_features.dim() > 1:
            task_features = task_features.squeeze(0)  # 去除批次维度
            
        # 处理任务嵌入维度
        if task_features.shape[0] != expected_dim:
            if self.debug_mode:
                print(f"Debug - 需要调整任务嵌入维度 从 {task_features.shape[0]} 到 {expected_dim}")
            
            # 如果维度过大，取平均池化或截断
            if task_features.shape[0] > expected_dim:
                task_features = task_features[:expected_dim]
            # 如果维度过小，填充0
            else:
                padding = torch.zeros(expected_dim - task_features.shape[0])
                task_features = torch.cat([task_features, padding])
        
        # 设备资源特征（归一化）
        device_features = torch.tensor([
            device.cpu_freq / 2e9,  # 归一化CPU频率
            device.remaining_energy / device.initial_energy if hasattr(device, 'remaining_energy') and hasattr(device, 'initial_energy') else 0.5,  # 归一化剩余能量
            device.data_rate / 20e6,  # 归一化数据传输速率
            device.current_load / device.cpu_freq,  # 归一化当前负载
        ]).float()
        
        # 添加服务器负载信息
        server_features = []
        for server in self.servers:
            # 归一化的服务器负载
            server_load = server.current_load / server.cpu_freq
            server_features.append(server_load)
        
        # 转为tensor
        server_features_tensor = torch.tensor(server_features).float()
        
        # 组合成最终状态向量
        state = torch.cat([
            task_features,            # 任务特征
            device_features,          # 设备特征
            server_features_tensor    # 服务器负载特征
        ])
        
        if self.debug_mode:
            print(f"Debug - 最终状态向量形状: {state.shape}")
            print(f"Debug - 服务器负载特征: {server_features}")
            
        return state
    
    def _execute_locally(self, device, task) -> Tuple[float, float]:
        """在本地设备上执行任务"""
        # 计算执行时间
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
    
    def _execute_offload(self, device, server, task) -> Tuple[float, float]:
        """卸载任务到边缘服务器执行，增强负载均衡考量"""
        # 计算传输时间 - 使用input_data_size或data_size
        transmission_time = task.input_data_size / server.bandwidth
        
        # 使用固定的网络延迟而不是随机值，减少能耗计算的波动
        network_latency = 0.02  # 固定为20毫秒的网络延迟
        transmission_time += network_latency
        
        # 计算执行时间 - 考虑服务器负载
        # 如果服务器负载较高，执行时间会适当增加
        load_factor = 1.0 + (server.current_load / server.cpu_freq) * 0.5  # 负载影响因子
        execution_time = (task.computing_cycles / server.cpu_freq) * load_factor
        
        # 计算总时间（传输+执行）
        total_time = transmission_time + execution_time
        
        # 计算传输能耗
        transmission_energy = device.transmit_power * transmission_time
        
        # 计算计算能耗
        computing_energy = server.energy_coefficient * (server.cpu_freq ** 2) * execution_time
        
        # 总能耗 - 添加上下限约束，避免极端值
        total_energy = transmission_energy + computing_energy
        
        # 限制能耗在合理范围内，避免极端值
        if total_energy < 1e-6:
            total_energy = 1e-6
        elif total_energy > 1e6:
            total_energy = 1e6
        
        # 根据服务器当前负载情况更新负载，避免过度集中
        # 计算负载增量
        load_increment = task.computing_cycles
        
        # 更新服务器负载
        server.current_load += load_increment
        
        # 打印详细的服务器负载更新信息
        if self.debug_mode:
            load_percentage = server.current_load / server.cpu_freq * 100
            print(f"GNN-DRL - 服务器 {server.id} 负载已更新: {server.current_load:.2e}/{server.cpu_freq:.2e} (负载比例={load_percentage:.2f}%)")
        
        return total_time, total_energy
    
    def _calculate_load_variance(self) -> float:
        """计算负载均衡度 - 使用统一的评估器"""
        return self.evaluator.calculate_load_balance(self.servers)
    
    def find_device(self, task):
        """查找任务所属的设备"""
        for device in self.devices:
            if task in device.task_queue:
                return device
        return None
    
    def _select_action(self, state):
        """使用DRL代理选择动作，增强负载均衡能力"""
        # 准备服务器负载信息供DRL代理使用
        server_loads = {}
        max_load = max([server.current_load for server in self.servers], default=0)
        max_load = max(max_load, 1e-6)  # 避免除零错误
        
        for i, server in enumerate(self.servers):
            # 计算服务器负载比例
            load_percentage = server.current_load / server.cpu_freq
            server_loads[i] = load_percentage
            
            # 打印服务器负载信息
            if self.debug_mode:
                print(f"GNN-DRL - 服务器 {i} 负载比例: {load_percentage:.4f}")
        
        with torch.no_grad():
            # 使用改进后的select_action方法，将服务器负载信息传入
            action = self.drl_agent.select_action(state.float(), server_loads)
            
            # 打印选择的动作
            if self.debug_mode:
                if action == 0:
                    print(f"GNN-DRL - 选择本地执行")
                else:
                    print(f"GNN-DRL - 选择卸载到服务器 {action-1}")
            
        return action

    def _load_pretrained_models(self):
        # 实现加载预训练模型的逻辑
        pass
