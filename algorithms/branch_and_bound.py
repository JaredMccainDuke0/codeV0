from typing import List, Tuple, Dict, Set, Any
import numpy as np
import heapq
from models.task import Task
from models.device import Device
from models.server import Server
from utils.evaluator import Evaluator

class BranchAndBoundAlgorithm:
    """
    基于分支定界法的任务调度算法，用于生成专家策略
    算法1的具体实现
    """
    def __init__(self, devices: List[Device], servers: List[Server], tasks: List[Task], config=None):
        """
        分支定界算法实现
        
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
        
        # 使用统一的评估器
        self.evaluator = Evaluator()
        
        # 权重系数，用于计算成本函数
        self.alpha = 1/3  # 延迟权重
        self.beta = 1/3   # 能耗权重
        self.gamma = 1/3  # 负载均衡权重
        
        # 当前最优解
        self.best_solution = None
        self.best_cost = float('inf')
        
        # 缓存数据
        self.cache = {}
        for server in self.servers:
            self.cache[server.id] = {}
            
        # 任务依赖图
        self.dependencies = self._build_dependencies()
        
        # 任务分配结果
        self.task_assignments = {}  # {task_id: (device_id or server_id, is_offloaded)}
        
        # 统计信息
        self.total_nodes = 0
        self.pruned_nodes = 0
        
    def _build_dependencies(self) -> Dict[int, List[int]]:
        """构建任务依赖图"""
        dependencies = {}
        for i, task in enumerate(self.tasks):
            dependencies[i] = []
            for j, other_task in enumerate(self.tasks):
                if task.id in other_task.dependencies:
                    dependencies[i].append(j)
        return dependencies
    
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
    
    def execute(self) -> Tuple[float, float, float]:
        """
        执行Branch and Bound算法，找到最优任务分配策略
        返回：(总完成时间, 总能耗, 负载均衡度)
        """
        # 初始化状态
        initial_state = {
            'allocated': [],  # 已分配的任务
            'device_loads': {device.id: device.current_load for device in self.devices},
            'server_loads': {server.id: server.current_load for server in self.servers},
            'total_delay': 0,
            'total_energy': 0,
            'decisions': {}  # 记录每个任务的决策 {task_idx: (device_idx, server_idx 或 None)}
        }
        
        # 初始化优先队列
        # 队列元素为 (下界, 状态id, 状态)
        priority_queue = [(0, 0, initial_state)]
        state_counter = 1
        
        # 获取任务的拓扑排序
        task_order = self._topological_sort()
        
        # 执行Branch and Bound
        while priority_queue:
            bound, _, state = heapq.heappop(priority_queue)
            
            # 如果下界大于当前最优解，则剪枝
            if bound >= self.best_cost:
                continue
                
            # 检查是否为完整解决方案
            if len(state['allocated']) == len(self.tasks):
                # 计算最终成本
                final_cost = self._calculate_cost(state)
                if final_cost < self.best_cost:
                    self.best_cost = final_cost
                    self.best_solution = state['decisions'].copy()
                continue
                
            # 获取下一个要分配的任务
            next_task_idx = None
            for task_idx in task_order:
                if task_idx not in state['allocated']:
                    # 检查依赖是否满足
                    deps_satisfied = True
                    for dep_idx in self.dependencies[task_idx]:
                        if dep_idx not in state['allocated']:
                            deps_satisfied = False
                            break
                    if deps_satisfied:
                        next_task_idx = task_idx
                        break
            
            if next_task_idx is None:
                continue  # 没有可分配的任务
                
            task = self.tasks[next_task_idx]
            
            # 为当前任务生成所有可能的分配方案
            for device_idx, device in enumerate(self.devices):
                # 本地执行
                local_state = self._create_new_state(state, next_task_idx, device_idx, None)
                if local_state:
                    local_bound = self._calculate_bound(local_state)
                    if local_bound < self.best_cost:
                        heapq.heappush(priority_queue, (local_bound, state_counter, local_state))
                        state_counter += 1
                
                # 卸载到服务器
                for server_idx, server in enumerate(self.servers):
                    # 检查是否可以从缓存中复用结果
                    reuse = self._check_cache_reuse(task, server)
                    offload_state = self._create_new_state(state, next_task_idx, device_idx, server_idx, reuse)
                    if offload_state:
                        offload_bound = self._calculate_bound(offload_state)
                        if offload_bound < self.best_cost:
                            heapq.heappush(priority_queue, (offload_bound, state_counter, offload_state))
                            state_counter += 1
        
        # 应用最优解
        if self.best_solution:
            self._apply_solution()
            
        # 计算并返回最终结果
        total_delay = 0
        total_energy = 0
        for task_idx, (device_idx, server_idx) in self.best_solution.items():
            device = self.devices[device_idx]
            task = self.tasks[task_idx]
            
            if server_idx is None:
                # 本地执行
                delay, energy = self._calculate_local_execution(device, task)
            else:
                # 卸载执行
                server = self.servers[server_idx]
                delay, energy = self._calculate_offload_execution(device, server, task)
                
            total_delay = max(total_delay, delay)  # 取最大完成时间
            total_energy += energy
            
        load_balance = self.calculate_load_balance()
        
        return total_delay, total_energy, load_balance
        
    def _create_new_state(self, state, task_idx, device_idx, server_idx, reuse=False):
        """创建新状态"""
        new_state = {
            'allocated': state['allocated'].copy(),
            'device_loads': state['device_loads'].copy(),
            'server_loads': state['server_loads'].copy(),
            'total_delay': state['total_delay'],
            'total_energy': state['total_energy'],
            'decisions': state['decisions'].copy()
        }
        
        device = self.devices[device_idx]
        task = self.tasks[task_idx]
        
        # 更新决策记录
        new_state['decisions'][task_idx] = (device_idx, server_idx)
        
        # 更新已分配任务列表
        new_state['allocated'].append(task_idx)
        
        if server_idx is None:
            # 本地执行
            delay, energy = self._calculate_local_execution(device, task)
            # 更新设备负载
            new_state['device_loads'][device.id] += task.computing_cycles
        else:
            # 卸载到服务器
            server = self.servers[server_idx]
            
            if reuse:
                # 使用缓存结果
                delay, energy = self._calculate_cache_reuse(device, server, task)
            else:
                # 正常卸载执行
                delay, energy = self._calculate_offload_execution(device, server, task)
                # 更新服务器负载
                new_state['server_loads'][server.id] += task.computing_cycles
        
        # 更新总延迟和能耗
        new_state['total_delay'] = max(new_state['total_delay'], delay)
        new_state['total_energy'] += energy
        
        return new_state
    
    def _calculate_bound(self, state):
        """计算状态的下界"""
        # 已知的成本
        known_cost = self._calculate_cost(state)
        
        # 剩余任务的估计成本（乐观估计）
        remaining_tasks = [i for i in range(len(self.tasks)) if i not in state['allocated']]
        estimated_delay = 0
        estimated_energy = 0
        
        for task_idx in remaining_tasks:
            task = self.tasks[task_idx]
            min_delay = float('inf')
            min_energy = float('inf')
            
            # 找出每个任务的最优执行方式（本地或卸载）
            for device in self.devices:
                local_delay, local_energy = self._calculate_local_execution(device, task)
                if local_delay < min_delay:
                    min_delay = local_delay
                if local_energy < min_energy:
                    min_energy = local_energy
                    
                for server in self.servers:
                    offload_delay, offload_energy = self._calculate_offload_execution(device, server, task)
                    if offload_delay < min_delay:
                        min_delay = offload_delay
                    if offload_energy < min_energy:
                        min_energy = offload_energy
            
            estimated_delay = max(estimated_delay, min_delay)
            estimated_energy += min_energy
        
        # 估计负载均衡的下界（乐观估计为最佳均衡）
        estimated_balance = 0  # 理想情况下完全均衡
        
        # 总成本下界
        bound = self.alpha * (state['total_delay'] + estimated_delay) + \
                self.beta * (state['total_energy'] + estimated_energy) + \
                self.gamma * estimated_balance
                
        return bound
    
    def _calculate_cost(self, state):
        """计算状态的成本函数值"""
        # 计算负载方差
        load_variance = self._calculate_load_variance(state)
        
        # 成本函数 C(s) = αTdelay + βEenergy + γ · Var(Le)
        cost = self.alpha * state['total_delay'] + \
               self.beta * state['total_energy'] + \
               self.gamma * load_variance
               
        return cost
    
    def _calculate_load_variance(self, state):
        """计算负载方差"""
        loads = []
        for device_id, load in state['device_loads'].items():
            loads.append(load)
        for server_id, load in state['server_loads'].items():
            loads.append(load)
            
        if not loads:
            return 0
            
        # 计算Jain公平性指数(JFI)
        n = len(loads)
        if n == 0 or sum(loads) == 0:
            return 0
            
        # 归一化负载
        max_load = max(loads) if max(loads) > 0 else 1
        normalized_loads = [load / max_load for load in loads]
        
        # 计算JFI
        numerator = sum(normalized_loads) ** 2
        denominator = n * sum([load ** 2 for load in normalized_loads])
        
        if denominator == 0:
            jfi = 1  # 避免除以零
        else:
            jfi = numerator / denominator
            
        # 将JFI映射到[0,100]范围，其中0表示最不平衡，100表示完全平衡
        if n == 1:
            balance = 100  # 只有一个节点时，负载自然是平衡的
        else:
            balance = (jfi - 1/n) / (1 - 1/n) * 100
            
        # 返回不平衡度作为方差（100-balance）
        return 100 - balance
    
    def calculate_load_balance(self) -> float:
        """计算负载均衡度 - 使用统一的评估器"""
        return self.evaluator.calculate_load_balance(self.servers)
    
    def _calculate_local_execution(self, device, task) -> Tuple[float, float]:
        """计算本地执行的延迟和能耗"""
        # 计算执行时间
        execution_time = task.computing_cycles / device.cpu_freq
        
        # 计算能耗
        computing_energy = device.energy_coefficient * (device.cpu_freq ** 2) * execution_time
        
        # 根据任务计算量调整能耗
        task_complexity_factor = task.computing_cycles / 1e6
        
        # 计算总能耗
        total_energy = computing_energy * task_complexity_factor * 1e9
        
        return execution_time, total_energy
    
    def _calculate_offload_execution(self, device, server, task) -> Tuple[float, float]:
        """计算卸载执行的延迟和能耗"""
        # 计算传输时间
        data_size = task.input_data_size
        transmission_time = data_size / device.data_rate
        
        # 计算在服务器上的执行时间
        execution_time = task.computing_cycles / server.cpu_freq
        
        # 计算结果返回时间
        result_transmission_time = task.output_data_size / device.data_rate
        
        # 总延迟
        total_delay = transmission_time + execution_time + result_transmission_time
        
        # 计算传输能耗
        transmission_energy = device.tx_power * transmission_time
        result_transmission_energy = device.tx_power * result_transmission_time
        
        # 计算总能耗
        total_energy = (transmission_energy + result_transmission_energy) * 1e9
        
        return total_delay, total_energy
    
    def _calculate_cache_reuse(self, device, server, task) -> Tuple[float, float]:
        """计算使用缓存复用结果的延迟和能耗"""
        # 只需要计算结果传输时间和能耗
        result_transmission_time = task.output_data_size / device.data_rate
        result_transmission_energy = device.tx_power * result_transmission_time
        
        # 总延迟和能耗
        total_delay = result_transmission_time
        total_energy = result_transmission_energy * 1e9
        
        return total_delay, total_energy
    
    def _check_cache_reuse(self, task, server) -> bool:
        """检查是否可以从缓存中复用结果"""
        # 简单实现，实际应该使用A-LSH和W-KNN算法
        cache_key = self._get_cache_key(task)
        server_cache = self.cache.get(server.id, {})
        
        return cache_key in server_cache
    
    def _get_cache_key(self, task) -> str:
        """获取任务的缓存键"""
        # 简单实现，实际应该基于任务特征
        return f"task_{task.id}_{task.computing_cycles}_{task.input_data_size}"
    
    def _apply_solution(self):
        """应用最优解决方案"""
        # 重置设备和服务器负载
        for device in self.devices:
            device.current_load = 0
        for server in self.servers:
            server.current_load = 0
            
        # 按照最优解分配任务
        for task_idx, (device_idx, server_idx) in self.best_solution.items():
            device = self.devices[device_idx]
            task = self.tasks[task_idx]
            
            if server_idx is None:
                # 本地执行
                device.current_load += task.computing_cycles
            else:
                # 卸载到服务器
                server = self.servers[server_idx]
                server.current_load += task.computing_cycles
                
    def get_expert_data(self) -> List[Dict]:
        """
        获取专家策略数据，用于训练GNN-IL模型
        返回：专家决策数据列表
        """
        if not self.best_solution:
            self.execute()  # 如果还没有执行算法，先执行
            
        expert_data = []
        
        for task_idx, (device_idx, server_idx) in self.best_solution.items():
            task = self.tasks[task_idx]
            device = self.devices[device_idx]
            
            # 提取特征
            features = {
                'task_id': task.id,
                'computing_cycles': task.computing_cycles,
                'input_data_size': task.input_data_size,
                'output_data_size': task.output_data_size,
                'dependencies': task.dependencies,
                'device_id': device.id,
                'device_cpu_freq': device.cpu_freq,
                'device_energy_coef': device.energy_coefficient,
                'device_tx_power': device.tx_power,
                'device_data_rate': device.data_rate,
                'device_current_load': device.current_load
            }
            
            # 提取标签
            if server_idx is None:
                # 本地执行
                label = {
                    'offload': 0,  # 0表示本地执行
                    'server_id': -1,
                    'resource_allocation': 0,
                    'bandwidth_allocation': 0
                }
            else:
                # 卸载到服务器
                server = self.servers[server_idx]
                label = {
                    'offload': 1,  # 1表示卸载执行
                    'server_id': server.id,
                    'resource_allocation': 1.0,  # 简化实现，实际应该是分配的计算资源比例
                    'bandwidth_allocation': 1.0  # 简化实现，实际应该是分配的带宽比例
                }
                
            expert_data.append({
                'features': features,
                'label': label
            })
            
        return expert_data #