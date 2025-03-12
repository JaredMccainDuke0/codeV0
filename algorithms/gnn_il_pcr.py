import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Set
import random
from models.task import Task
from models.device import Device
from models.server import Server
import networkx as nx
import time

# 导入LSH相关依赖
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial.distance import cosine, euclidean

# 导入统一评估器
from utils.evaluator import Evaluator

class GNNILPCR:
    def __init__(self, devices: List, servers: List, tasks: List, config=None):
        """
        基于GNN和模仿学习的任务卸载算法，集成部分计算重用功能
        
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
        
        # 添加调试模式标志，默认为False
        self.debug_mode = False
        
        # 图神经网络模型
        self.gnn = self._build_gnn_model()
        
        # 模仿学习模型
        self.il_model = self._build_il_model()
        
        # 任务队列和依赖图
        self.task_queue = []
        self.dependencies = {}
        
        # 结果缓存
        self.cache = {}
        for server in self.servers:
            self.cache[server.id] = {
                'items': {},      # 缓存项 {cache_key: {'features': [...], 'result': ..., 'score': ..., 'timestamp': ...}}
                'max_size': 100,  # 最大缓存大小
                'lsh_index': None,# LSH索引
                'feature_dim': 10,# 特征维度
            }
            # 初始化A-LSH索引
            self._init_lsh_index(server.id)
            
        # 统计数据
        self.total_tasks = 0
        self.cache_hits = 0
        self.offloaded_tasks = 0
        self.local_tasks = 0
        
        # 使用统一的评估器
        self.evaluator = Evaluator()
        
        # 加载预训练模型（如果存在）
        self._load_pretrained_models()
        
    def _build_gnn_model(self):
        """构建GNN模型"""
        class GNN(nn.Module):
            def __init__(self, input_dim=10, hidden_dim=64, output_dim=32):
                super(GNN, self).__init__()
                self.conv1 = nn.Linear(input_dim, hidden_dim)
                self.conv2 = nn.Linear(hidden_dim, hidden_dim)
                self.conv3 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x, edge_index):
                # 简化的图卷积实现
                h = F.relu(self.conv1(x))
                h = F.relu(self.conv2(h))
                h = self.conv3(h)
                return h
                
        return GNN()
        
    def _build_il_model(self):
        """构建模仿学习模型"""
        class ILModel(nn.Module):
            def __init__(self, input_dim=32, hidden_dim=64):
                super(ILModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                
                # 卸载决策头（分类）
                self.offload_head = nn.Linear(hidden_dim, 2)  # 2类：本地执行 or 卸载
                
                # 服务器选择头（分类）
                self.server_head = nn.Linear(hidden_dim, 10)  # 假设最多10个服务器
                
                # 资源分配头（回归）
                self.resource_head = nn.Linear(hidden_dim, 1)
                
                # 带宽分配头（回归）
                self.bandwidth_head = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                h = F.relu(self.fc1(x))
                h = F.relu(self.fc2(h))
                
                offload_pred = F.softmax(self.offload_head(h), dim=1)
                server_pred = F.softmax(self.server_head(h), dim=1)
                resource_pred = torch.sigmoid(self.resource_head(h))
                bandwidth_pred = torch.sigmoid(self.bandwidth_head(h))
                
                return offload_pred, server_pred, resource_pred, bandwidth_pred
                
        return ILModel()
        
    def _load_pretrained_models(self):
        """加载预训练模型"""
        try:
            if os.path.exists("models/gnn_il_model.pth"):
                self.gnn.load_state_dict(torch.load("models/gnn_il_model.pth"))
                print("已加载GNN预训练模型")
                
            if os.path.exists("models/il_model.pth"):
                self.il_model.load_state_dict(torch.load("models/il_model.pth"))
                print("已加载IL预训练模型")
        except Exception as e:
            print(f"加载预训练模型失败 {e}")
    
    def _build_task_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建任务依赖图
        返回：节点特征和边索引
        """
        # 初始化图数据
        num_tasks = len(self.tasks)
        node_features = torch.zeros((num_tasks, 10))  # 假设每个任务有10个特征
        edge_index = []
        
        # 创建任务ID到索引的映射，避免内部循环
        task_id_to_idx = {task.id: i for i, task in enumerate(self.tasks)}
        
        # 提取任务特征和依赖关系
        for i, task in enumerate(self.tasks):
            # 归一化特征
            computing_cycles_norm = task.computing_cycles / 1e9
            input_data_size_norm = task.input_data_size / 1e6
            output_data_size_norm = task.output_data_size / 1e6
            
            # 构建特征向量
            node_features[i] = torch.tensor([
                computing_cycles_norm,
                input_data_size_norm,
                output_data_size_norm,
                len(task.dependencies) / 10,  # 归一化依赖数
                task.id / 1000,  # 归一化任务ID
                0.5,  # 占位特征
                0.5,  # 占位特征
                0.5,  # 占位特征
                0.5,  # 占位特征
                0.5,  # 占位特征
            ])
            
            # 使用映射，避免内部循环
            for dep_id in task.dependencies:
                if dep_id in task_id_to_idx:
                    j = task_id_to_idx[dep_id]
                    edge_index.append([j, i])  # 从依赖指向当前任务
        # 转换为Tensor
        if edge_index:
            edge_index = torch.tensor(edge_index).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        return node_features, edge_index
    
    def _init_lsh_index(self, server_id):
        """初始化A-LSH索引"""
        feature_dim = self.cache[server_id]['feature_dim']
        # 使用高斯随机投影作为LSH
        self.cache[server_id]['lsh_index'] = {
            'projections': np.random.randn(10, feature_dim),  # 10个哈希函数
            'buckets': {},  # 存储桶 {bucket_key: [cache_keys]}
            'threshold': 0.5  # 哈希阈值
        }
    
    def _extract_task_features(self, task) -> np.ndarray:
        """提取任务特征向量"""
        # 提取关键特征并归一化
        computing_cycles_norm = task.computing_cycles / 1e9
        input_data_size_norm = task.input_data_size / 1e6
        output_data_size_norm = task.output_data_size / 1e6
        
        # 构建特征向量（与_build_task_graph中保持一致）
        features = np.array([
            computing_cycles_norm,
            input_data_size_norm, 
            output_data_size_norm,
            len(task.dependencies) / 10,
            task.id / 1000,
            0.5, 0.5, 0.5, 0.5, 0.5
        ])
        
        return features
    
    def _compute_lsh_hash(self, server_id, features) -> str:
        """计算特征的LSH哈希"""
        projections = self.cache[server_id]['lsh_index']['projections']
        threshold = self.cache[server_id]['lsh_index']['threshold']
        
        # 计算投影
        projections_result = np.dot(projections, features)
        
        # 二值化结果
        binary_hash = (projections_result > threshold).astype(int)
        
        # 转换为字符串哈希
        hash_str = ''.join(str(bit) for bit in binary_hash)
        
        return hash_str
    
    def _a_lsh_search(self, server_id, task_features) -> List[str]:
        """A-LSH粗略搜索，返回候选缓存项的键"""
        # 计算查询特征的哈希
        hash_str = self._compute_lsh_hash(server_id, task_features)
        
        # 获取匹配的桶
        buckets = self.cache[server_id]['lsh_index']['buckets']
        candidates = buckets.get(hash_str, [])
        
        return candidates
    
    def _w_knn_search(self, server_id, task_features, candidates, k=3) -> List[Tuple[str, float]]:
        """W-KNN精细搜索，计算加权距离并返回前K个最相似的缓存项"""
        if not candidates:
            return []
            
        # 计算到每个候选项的距离
        distances = []
        for cache_key in candidates:
            if cache_key in self.cache[server_id]['items']:
                cache_item = self.cache[server_id]['items'][cache_key]
                cache_features = cache_item['features']
                
                # 计算余弦距离（相似度）
                dist = cosine(task_features, cache_features)
                
                # 计算权重（考虑时间衰减和缓存分数）
                current_time = time.time()
                time_diff = current_time - cache_item['timestamp']
                time_weight = np.exp(-0.01 * time_diff)  # 时间衰减因子
                
                weighted_dist = dist * (1 - time_weight * cache_item['score'])
                
                distances.append((cache_key, weighted_dist))
        
        # 按加权距离排序并返回前K个
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def execute(self) -> Tuple[float, float, float]:
        """
        执行GNN-IL-PCR算法
        返回：总完成时间 总能耗 负载均衡度
        """
        # 构建任务依赖图
        node_features, edge_index = self._build_task_graph()
        
        # 使用GNN编码任务结构
        with torch.no_grad():
            task_embeddings = self.gnn(node_features, edge_index)
        
        # 全局卸载决策
        offload_decisions = []
        server_decisions = []
        resource_allocations = []
        bandwidth_allocations = []
        
        # 添加负载均衡考量 - 跟踪服务器负载
        server_loads = {server.id: server.current_load for server in self.servers}
        
        for i, task in enumerate(self.tasks):
            # 获取任务嵌入
            task_embedding = task_embeddings[i].unsqueeze(0)
            
            # 预测决策
            with torch.no_grad():
                offload_pred, server_pred, resource_pred, bandwidth_pred = self.il_model(task_embedding)
            
            # 解析预测结果
            offload = torch.argmax(offload_pred, dim=1).item()  # 0:本地, 1:卸载
            
            # 修改服务器选择逻辑，增加负载均衡考量
            if offload == 1:
                # 获取模型预测的服务器索引
                model_server_idx = torch.argmax(server_pred, dim=1).item()
                
                # 检查该服务器的负载情况
                if model_server_idx < len(self.servers):
                    model_server = self.servers[model_server_idx]
                    model_server_load = server_loads.get(model_server.id, 0) / model_server.cpu_freq
                    
                    # 如果预测服务器负载过高，寻找负载较低的替代服务器
                    if model_server_load > 0.7:  # 70%负载阈值
                        # 寻找负载最低的服务器
                        alt_server_idx = -1
                        min_load = float('inf')
                        for j, server in enumerate(self.servers):
                            load_ratio = server_loads.get(server.id, 0) / server.cpu_freq
                            if load_ratio < min_load:
                                min_load = load_ratio
                                alt_server_idx = j
                        
                        # 如果找到了更好的服务器，使用
                        if alt_server_idx >= 0 and min_load < model_server_load:
                            server_idx = alt_server_idx
                        else:
                            server_idx = model_server_idx
                    else:
                        server_idx = model_server_idx
                else:
                    # 模型预测的服务器索引无效，使用负载最低的服务器
                    server_idx = 0
                    min_load = float('inf')
                    for j, server in enumerate(self.servers):
                        load_ratio = server_loads.get(server.id, 0) / server.cpu_freq
                        if load_ratio < min_load:
                            min_load = load_ratio
                            server_idx = j
            else:
                server_idx = -1  # 本地执行
            
            resource = resource_pred.item() if offload == 1 else 0
            bandwidth = bandwidth_pred.item() if offload == 1 else 0
            
            # 保存决策
            offload_decisions.append(offload)
            server_decisions.append(server_idx)
            resource_allocations.append(resource)
            bandwidth_allocations.append(bandwidth)
            
            # 更新服务器负载预测
            if offload == 1 and server_idx >= 0 and server_idx < len(self.servers):
                server = self.servers[server_idx]
                server_loads[server.id] = server_loads.get(server.id, 0) + task.computing_cycles
        
        # 拓扑排序处理任务依赖
        execution_order = self._topological_sort()
        
        # 执行任务
        total_completion_time = 0
        total_energy = 0
        
        for idx in execution_order:
            task = self.tasks[idx]
            offload = offload_decisions[idx]
            server_idx = server_decisions[idx]
            
            if offload == 0:
                # 本地执行
                device = self.find_device(task)
                delay, energy = self._execute_locally(device, task)
                self.local_tasks += 1
            else:
                # 卸载执行
                device = self.find_device(task)
                
                if server_idx >= 0 and server_idx < len(self.servers):
                    server = self.servers[server_idx]
                    
                    # 缓存复用检查
                    task_features = self._extract_task_features(task)
                    
                    # A-LSH粗略搜索
                    candidates = self._a_lsh_search(server.id, task_features)
                    
                    # W-KNN精细搜索
                    similar_items = self._w_knn_search(server.id, task_features, candidates)
                    
                    # 检查最相似项是否满足复用条件
                    if similar_items and similar_items[0][1] < 0.2:  # 距离阈值
                        # 复用缓存结果
                        cache_key = similar_items[0][0]
                        delay, energy = self._use_cached_result(device, server, task)
                        
                        # 更新缓存分数
                        self._update_cache_score(server.id, cache_key)
                        self.cache_hits += 1
                    else:
                        # 正常卸载执行
                        delay, energy = self._execute_offload(device, server, task)
                        
                        # 缓存结果
                        self._cache_result(task, server)
                        self.offloaded_tasks += 1
                else:
                    # 服务器索引无效，回退到本地执行
                    device = self.find_device(task)
                    delay, energy = self._execute_locally(device, task)
                    self.local_tasks += 1
            
            # 更新总延迟和能耗
            total_completion_time = max(total_completion_time, delay)
            total_energy += energy
            
            self.total_tasks += 1
        
        # 获取任务处理完成后的结果
        self.total_tasks = len(self.tasks)
        
        # 计算平均完成时间和总能耗
        avg_completion_time = total_completion_time / max(1, self.total_tasks)
        
        # 计算负载均衡度
        balance = self._calculate_load_balance()
        
        # 只在调试模式下打印详细统计信息
        server_loads = [server.current_load for server in self.servers]
        if self.debug_mode:
            print(f"负载: {server_loads}")
            print(f"负载均衡度: {balance:.2f}%")
            print(f"总任务数: {self.total_tasks}, 缓存命中率: {self.cache_hit_rate():.2f}%, 本地任务: {self.local_tasks}, 卸载任务: {self.offloaded_tasks}")
        
        return avg_completion_time, total_energy, balance
    
    def _topological_sort(self) -> List[int]:
        """对任务进行拓扑排序"""
        # 计算入度
        in_degree = {i: 0 for i in range(len(self.tasks))}
        graph = {i: [] for i in range(len(self.tasks))}
        
        # 构建图和计算入度
        for i, task in enumerate(self.tasks):
            for dep_id in task.dependencies:
                for j, other_task in enumerate(self.tasks):
                    if other_task.id == dep_id:
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
    
    def _execute_locally(self, device, task) -> Tuple[float, float]:
        """在本地设备上执行任务"""
        # 计算执行时间
        execution_time = task.computing_cycles / device.cpu_freq
        
        # 计算能耗
        computing_energy = device.energy_coefficient * (device.cpu_freq ** 2) * execution_time
        
        # 根据任务计算量调整能耗
        task_complexity_factor = task.computing_cycles / 1e6
        
        # 计算总能耗 - 与其他算法保持一致
        total_energy = computing_energy * task_complexity_factor * 1e6
        
        # 更新设备负载
        device.current_load += task.computing_cycles
        
        return execution_time, total_energy
    
    def _execute_offload(self, device, server, task) -> Tuple[float, float]:
        """卸载任务到边缘服务器执行"""
        # 计算传输时间
        transmission_time = task.input_data_size / server.bandwidth
        
        # 使用固定的网络延迟而不是随机值，减少能耗计算的波动
        network_latency = 0.02  # 固定20毫秒的网络延迟
        transmission_time += network_latency
        
        # 计算执行时间
        execution_time = task.computing_cycles / server.cpu_freq
        
        # 计算总时间
        total_time = transmission_time + execution_time
        
        # 计算传输能耗
        transmission_energy = device.transmit_power * transmission_time
        
        # 计算计算能耗
        computing_energy = server.energy_coefficient * (server.cpu_freq ** 2) * execution_time
        
        # 总能耗 - 添加放大因子，与其他算法保持一致
        total_energy = (transmission_energy * 0.5 + computing_energy * 0.7) * 1e6
        
        # 限制能耗在合理范围内，避免极端情况
        if total_energy < 1e-6:
            total_energy = 1e-6
        elif total_energy > 1e6:
            total_energy = 1e6
        
        # 更新服务器负载
        server.current_load += task.computing_cycles
        
        return total_time, total_energy
    
    def _get_cache_key(self, task) -> str:
        """获取任务的缓存键"""
        # 生成一个唯一的缓存键，包含任务的关键特征
        return f"task_{task.id}_{task.computing_cycles}_{task.input_data_size}_{task.output_data_size}"
    
    def _cache_result(self, task, server):
        """缓存任务执行结果"""
        cache_key = self._get_cache_key(task)
        task_features = self._extract_task_features(task)
        
        # 模拟执行结果
        execution_result = {
            'output_data': f"result_for_task_{task.id}",
            'execution_time': task.computing_cycles / server.cpu_freq
        }
        
        # 缓存项
        cache_item = {
            'features': task_features,
            'result': execution_result,
            'score': 1.0,  # 初始分数
            'timestamp': time.time()
        }
        
        # 添加到缓存
        self.cache[server.id]['items'][cache_key] = cache_item
        
        # 添加到LSH索引
        hash_str = self._compute_lsh_hash(server.id, task_features)
        if hash_str not in self.cache[server.id]['lsh_index']['buckets']:
            self.cache[server.id]['lsh_index']['buckets'][hash_str] = []
        self.cache[server.id]['lsh_index']['buckets'][hash_str].append(cache_key)
        
        # 检查缓存大小并进行淘汰
        self._evict_cache_if_needed(server.id)
    
    def _update_cache_score(self, server_id, cache_key):
        """更新缓存项的分数和时间戳"""
        if cache_key in self.cache[server_id]['items']:
            # 添加一个时间衰减增益
            current_time = time.time()
            last_time = self.cache[server_id]['items'][cache_key]['timestamp']
            time_factor = np.exp(-0.01 * (current_time - last_time))
            
            # 更新分数（有上限1.0）
            current_score = self.cache[server_id]['items'][cache_key]['score']
            new_score = min(current_score + 0.1 * time_factor, 1.0)
            
            # 更新时间戳和分数
            self.cache[server_id]['items'][cache_key]['score'] = new_score
            self.cache[server_id]['items'][cache_key]['timestamp'] = current_time
    
    def _evict_cache_if_needed(self, server_id):
        """如果缓存超过容量，按LRU策略淘汰"""
        max_size = self.cache[server_id]['max_size']
        items = self.cache[server_id]['items']
        
        if len(items) > max_size:
            # 按最后访问时间排序
            sorted_keys = sorted(items.keys(), key=lambda k: items[k]['timestamp'])
            
            # 淘汰最旧的项目
            keys_to_evict = sorted_keys[:len(items) - max_size]
            
            for key in keys_to_evict:
                # 从LSH索引中移除
                features = items[key]['features']
                hash_str = self._compute_lsh_hash(server_id, features)
                if hash_str in self.cache[server_id]['lsh_index']['buckets']:
                    if key in self.cache[server_id]['lsh_index']['buckets'][hash_str]:
                        self.cache[server_id]['lsh_index']['buckets'][hash_str].remove(key)
                
                # 从缓存中移除
                del items[key]
    
    def _is_cache_valid(self, cache_entry, task) -> bool:
        """检查缓存项是否对当前任务有效"""
        # 在实际实现中，这里可以有更复杂的逻辑
        # 如检查任务特征是否与缓存项足够相似
        
        # 简单实现：如果计算量和数据大小相同，则认为有效
        if 'result' not in cache_entry:
            return False
            
        return True
    
    def _use_cached_result(self, device, server, task) -> Tuple[float, float]:
        """使用缓存的结"""
        # 只需要计算结果返回的时间和能耗
        result_transmission_time = task.output_data_size / device.data_rate
        result_transmission_energy = device.transmit_power * result_transmission_time
        
        # 总延迟和能耗 - 添加放大因子，与其他算法保持一致
        total_delay = result_transmission_time
        total_energy = result_transmission_energy * 0.3 * 1e6
        
        # 限制能耗在合理范围内，避免极端情况
        if total_energy < 1e-6:
            total_energy = 1e-6
        elif total_energy > 1e6:
            total_energy = 1e6
        
        return total_delay, total_energy
    
    def _calculate_load_balance(self) -> float:
        """计算负载均衡度 - 使用统一的评估器"""
        return self.evaluator.calculate_load_balance(self.servers)
        
    def find_device(self, task):
        """找到拥有该任务的设备"""
        for device in self.devices:
            if task in device.tasks:
                return device
        return self.devices[0]  # 默认返回第一个设备
    def cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        return (self.cache_hits / self.total_tasks) * 100 if self.total_tasks > 0 else 0
