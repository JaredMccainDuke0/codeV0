class ExperimentConfig:
    def __init__(self):
        # 网络环境参数
        self.area_size = (1000, 1000)  # 区域大小(m)
        self.num_devices = 50          # 终端设备数量
        self.num_servers = 10          # 边缘服务器数量
        self.server_coverage = 200     # 服务器覆盖半径(m)
        
        # 设备参数
        self.device_cpu_freq = (0.5, 1.5)  # CPU频率范围(GHz)
        self.device_power = (0.1, 1.0)     # 传输功率范围(W)
        self.energy_coeff = 1e-9          # 能耗系数，统一所有算法使用相同标准
        
        # 服务器参数
        self.server_cpu_freq = (3.0, 5.0)  # CPU频率范围(GHz)
        self.server_bandwidth = (1, 5)      # 带宽范围(MHz)
        
        # 任务参数
        self.num_task_types = 10           # 任务类型数量
        self.task_arrival_rate = 5         # 平均到达率(任务/秒)
        self.computing_cycles = (1e6, 1e9) # 计算量范围
        self.data_size = (0.5, 10)         # 数据大小范围(MB)
        self.num_subtasks = (5, 15)        # 子任务数量范围
        self.generate_dependencies = True   # 是否生成任务依赖
        self.distribute_servers = False     # 是否分布式放置服务器

        # GNN模型参数
        self.hidden_dim = 128
        self.n_heads = 8
        self.learning_rate = 0.001
        
        # 性能评估参数
        self.energy_scale = 1e10          # 能耗缩放因子，确保所有算法使用相同的缩放标准
        self.simulate_duration = 3600     # 模拟时长(秒)