class Server:
    """
    边缘服务器类，表示一个计算服务器
    
    属性:
        id (int): 服务器唯一标识
        cpu_freq (float): CPU频率（Hz）
        energy_coefficient (float): 能耗系数
        bandwidth (float): 带宽（bps）
    """
    
    def __init__(self, id, cpu_freq, energy_coefficient, bandwidth):
        """
        初始化服务器
        
        参数:
            id (int): 服务器唯一标识
            cpu_freq (float): CPU频率（Hz）
            energy_coefficient (float): 能耗系数
            bandwidth (float): 带宽（bps）
        """
        self.id = id
        self.cpu_freq = cpu_freq
        self.energy_coefficient = energy_coefficient
        self.bandwidth = bandwidth
        
        # 附加属性
        self.tasks = []          # 服务器上的任务列表
        self.current_load = 0    # 当前计算负载
        
    def add_task(self, task):
        """向服务器添加任务"""
        self.tasks.append(task)
        
    def clone(self):
        """创建服务器的深拷贝"""
        cloned = Server(
            id=self.id,
            cpu_freq=self.cpu_freq,
            energy_coefficient=self.energy_coefficient,
            bandwidth=self.bandwidth
        )
        
        # 复制任务列表和负载
        cloned.tasks = self.tasks.copy()
        cloned.current_load = self.current_load
        
        return cloned
    
    def __repr__(self):
        return f"Server(id={self.id}, cpu_freq={self.cpu_freq:.2e}Hz, bandwidth={self.bandwidth:.2e}bps, load={self.current_load:.2e})" 