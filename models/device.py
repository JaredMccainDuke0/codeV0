class Device:
    """
    终端设备类，表示一个计算设备
    
    属性:
        id (int): 设备唯一标识
        cpu_freq (float): CPU频率（Hz）
        energy_coefficient (float): 能耗系数
        tx_power (float): 传输功率（W）
        data_rate (float): 数据传输速率（bps）
    """
    
    def __init__(self, id, cpu_freq, energy_coefficient, tx_power, data_rate):
        """
        初始化设备
        
        参数:
            id (int): 设备唯一标识
            cpu_freq (float): CPU频率（Hz）
            energy_coefficient (float): 能耗系数
            tx_power (float): 传输功率（W）
            data_rate (float): 数据传输速率（bps）
        """
        self.id = id
        self.cpu_freq = cpu_freq
        self.energy_coefficient = energy_coefficient
        self.tx_power = tx_power
        self.data_rate = data_rate
        
        # 附加属性
        self.tasks = []           # 设备上的任务列表
        self.task_queue = []      # 兼容旧代码
        self.current_load = 0     # 当前计算负载
        self.transmit_power = tx_power  # 兼容旧代码
        self.remaining_energy = 100.0   # 设备剩余能量，默认为100单位
        
    def add_task(self, task):
        """向设备添加任务"""
        self.tasks.append(task)
        self.task_queue.append(task)  # 兼容旧代码
        
    def clone(self):
        """创建设备的深拷贝"""
        cloned = Device(
            id=self.id,
            cpu_freq=self.cpu_freq,
            energy_coefficient=self.energy_coefficient,
            tx_power=self.tx_power,
            data_rate=self.data_rate
        )
        
        # 复制任务列表
        cloned.tasks = self.tasks.copy()
        cloned.task_queue = self.task_queue.copy()  # 兼容旧代码
        cloned.current_load = self.current_load
        cloned.remaining_energy = self.remaining_energy
        
        return cloned
    
    def __repr__(self):
        return f"Device(id={self.id}, cpu_freq={self.cpu_freq:.2e}Hz, tx_power={self.tx_power}W, data_rate={self.data_rate:.2e}bps, tasks={len(self.tasks)})" 