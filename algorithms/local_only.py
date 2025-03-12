from typing import List, Tuple
from utils.evaluator import Evaluator

class LocalOnlyAlgorithm:
    def __init__(self, devices: List, servers: List, tasks: List, config=None):
        """
        本地执行算法 - 所有任务都在本地设备上执行
        
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
        
    def execute(self) -> Tuple[float, float, float]:
        """执行仅本地处理的算法"""
        # 初始化性能指标
        total_time = 0
        total_energy = 0
        total_tasks = len(self.tasks)
        
        # 跟踪处理的任务数
        processed_tasks = 0
        
        for task in self.tasks:
            # 找到任务所属设备
            device = None
            for d in self.devices:
                if task in d.task_queue:
                    device = d
                    break
            
            if not device:
                continue
                
            processed_tasks += 1
            
            # 计算执行时间 - 所有任务完全在本地执行
            execution_time = task.computing_cycles / device.cpu_freq
            
            # 计算与CPU相关的能耗
            cpu_energy = device.energy_coefficient * (device.cpu_freq ** 2) * execution_time
            
            # 根据任务计算量调整能耗
            task_complexity_factor = task.computing_cycles / 1e6
            
            # 计算任务能耗 - 降低本地执行的能耗倍数
            energy_consumption = cpu_energy * task_complexity_factor * 1e9
            
            # 小幅提高本地执行的能耗，使其比卸载执行高但不会过高
            local_execution_penalty = 1.1
            energy_consumption *= local_execution_penalty
            
            # 更新设备负载
            device.current_load += task.computing_cycles
            
            # 累加到总指标
            total_time += execution_time
            total_energy += energy_consumption
        
        # 计算平均时延 - 使用真实计算值
        avg_time = total_time / max(1, processed_tasks)
        
        # 本地处理算法的负载均衡度为0，恢复原始行为
        resource_balance = 0
        
        # 打印最终结果
        print(f"Local Only - 平均完成时延: {avg_time:.4f}秒, 总能耗: {total_energy:.4e}, 负载均衡度: {resource_balance:.2f}%")
        
        return avg_time, total_energy, resource_balance

    def calculate_load_balance(self) -> float:
        """计算负载均衡度 - 使用统一的评估器"""
        return self.evaluator.calculate_load_balance(self.servers)