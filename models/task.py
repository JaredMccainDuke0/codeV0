class Task:
    """
    任务类，表示一个计算任务
    
    属性:
        id (int): 任务唯一标识
        computing_cycles (float): 计算需要的CPU周期数
        input_data_size (float): 输入数据大小（字节）
        output_data_size (float): 输出数据大小（字节）
        dependencies (List[int]): 依赖任务的ID列表
    """
    
    def __init__(self, id, computing_cycles, input_data_size, output_data_size, dependencies=None):
        """
        初始化任务
        
        参数:
            id (int): 任务唯一标识
            computing_cycles (float): 计算需要的CPU周期数
            input_data_size (float): 输入数据大小（字节）
            output_data_size (float): 输出数据大小（字节）
            dependencies (List[int], optional): 依赖任务的ID列表
        """
        self.id = id
        self.computing_cycles = computing_cycles
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.dependencies = dependencies if dependencies is not None else []
        
        # 附加属性
        self.priority = 1.0  # 任务优先级
        self.subtasks = 0    # 子任务数量
        self.type = 0        # 任务类型（0=普通计算，1=数据密集型，2=计算密集型）
        self.data_size = input_data_size  # 兼容性属性
        
    def clone(self):
        """创建任务的深拷贝"""
        cloned = Task(
            id=self.id,
            computing_cycles=self.computing_cycles,
            input_data_size=self.input_data_size,
            output_data_size=self.output_data_size,
            dependencies=self.dependencies.copy()
        )
        cloned.priority = self.priority
        cloned.subtasks = self.subtasks
        cloned.type = self.type
        
        return cloned
    
    def __repr__(self):
        return f"Task(id={self.id}, cycles={self.computing_cycles:.2e}, in_size={self.input_data_size:.2e}, out_size={self.output_data_size:.2e}, deps={self.dependencies})" 