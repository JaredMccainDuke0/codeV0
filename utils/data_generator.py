import numpy as np
import random
from typing import List, Tuple
from models.task import Task
from models.device import Device
from models.server import Server

class DataGenerator:
    """数据生成器，用于生成设备、服务器和任务"""
    
    def __init__(self, config):
        """初始化数据生成器"""
        self.config = config
        
    def generate_devices(self) -> List[Device]:
        """生成移动设备列表"""
        devices = []
        
        for i in range(self.config.num_devices):
            # 随机生成设备的CPU频率和传输功率
            cpu_freq = random.uniform(1.0, 2.0) * 1e9  # 1-2 GHz
            tx_power = random.uniform(0.1, 0.5)  # 0.1-0.5 W
            energy_coefficient = random.uniform(1.0, 2.0) * 1e-27
            data_rate = random.uniform(5, 20) * 1e6  # 5-20 Mbps
            
            # 创建设备对象
            device = Device(
                id=i,
                cpu_freq=cpu_freq,
                energy_coefficient=energy_coefficient,
                tx_power=tx_power,
                data_rate=data_rate
            )
            
            devices.append(device)
            
        return devices
        
    def generate_servers(self) -> List[Server]:
        """生成边缘服务器列表"""
        servers = []
        
        for i in range(self.config.num_servers):
            # 随机生成服务器的CPU频率和带宽
            cpu_freq = random.uniform(5.0, 10.0) * 1e9  # 恢复为5-10 GHz
            bandwidth = random.uniform(50, 100) * 1e6  # 50-100 Mbps
            energy_coefficient = random.uniform(1.0, 2.0) * 1e-27
            
            # 创建服务器对象
            server = Server(
                id=i,
                cpu_freq=cpu_freq,
                energy_coefficient=energy_coefficient,
                bandwidth=bandwidth
            )
            
            servers.append(server)
            
        # 模拟不同地理位置的服务器（未来可根据实际地理分布修改）
        if self.config.distribute_servers:
            pass  # 未来可以添加地理分布逻辑
            
        return servers
    
    def generate_tasks(self) -> List[Task]:
        """生成任务列表"""
        tasks = []
        
        # 随机生成任务数量（基于泊松分布）
        num_tasks = np.random.poisson(self.config.task_arrival_rate)
        
        for i in range(num_tasks):
            # 随机生成任务类型、子任务数、计算量、数据大小和优先级
            task_type = random.randint(0, 2)  # 0=普通计算，1=数据密集型，2=计算密集型
            subtasks = random.randint(1, 10)
            
            # 根据任务类型调整计算量和数据大小
            if task_type == 0:  # 普通任务
                computing_cycles = random.uniform(1.0, 5.0) * 1e9  # 乘以10，由1e8变为1e9
                input_data_size = random.uniform(0.1, 1.0) * 1e6
                output_data_size = input_data_size * 0.2
            elif task_type == 1:  # 数据密集型
                computing_cycles = random.uniform(1.0, 3.0) * 1e9  # 乘以10，由1e8变为1e9
                input_data_size = random.uniform(1.0, 10.0) * 1e6
                output_data_size = input_data_size * 0.3
            else:  # 计算密集型
                computing_cycles = random.uniform(5.0, 10.0) * 1e9  # 乘以10，由1e8变为1e9
                input_data_size = random.uniform(0.1, 0.5) * 1e6
                output_data_size = input_data_size * 0.1
                
            priority = random.uniform(0.5, 1.0)
            
            # 创建任务对象
            task = Task(
                id=i,
                computing_cycles=computing_cycles,
                input_data_size=input_data_size,
                output_data_size=output_data_size
            )
            task.type = task_type
            task.subtasks = subtasks
            task.priority = priority
            
            # 随机生成任务依赖关系
            if self.config.generate_dependencies and i > 0:
                # 10%的概率生成依赖
                if random.random() < 0.1:
                    # 选择1-3个已有任务作为依赖
                    num_deps = random.randint(1, min(3, i))
                    deps = random.sample(range(i), num_deps)
                    task.dependencies = deps
            
            tasks.append(task)
            
        return tasks
    
    def assign_tasks_to_devices(self, devices: List[Device], tasks: List[Task]):
        """将任务分配给设备"""
        if not devices or not tasks:
            return
            
        # 随机分配任务给设备
        for task in tasks:
            # 随机选择一个设备
            device_idx = random.randint(0, len(devices) - 1)
            device = devices[device_idx]
            
            # 将任务分配给设备
            device.add_task(task)