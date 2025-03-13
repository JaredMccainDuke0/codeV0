import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os
from typing import List, Dict, Tuple
import matplotlib.font_manager as fm
import json
import matplotlib.ticker as ticker
import random
import pickle
import glob
from tqdm import tqdm

# 导入配置和工具
from config import ExperimentConfig
from utils.data_generator import DataGenerator

# 导入模型类
from models.task import Task
from models.device import Device
from models.server import Server

# 导入算法
from algorithms import LocalOnlyAlgorithm, GreedyAlgorithm, GNNDRL, GNNILPCR
from algorithms.branch_and_bound import BranchAndBoundAlgorithm

# 设置matplotlib显示中文
def setup_chinese_font():
    """检测并设置中文字体"""
    # 检查系统中可用的中文字体
    chinese_fonts = []
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 常见中文字体
    target_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'FangSong', 'STSong', 'NSimSun']
    
    for font in target_fonts:
        if font in all_fonts:
            chinese_fonts.append(font)
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts
        # 设置为True以便正确显示负号
        plt.rcParams['axes.unicode_minus'] = True
        print(f"已配置中文字体: {chinese_fonts[0]}")
        return True
    else:
        print("未找到中文字体，将使用英文标题")
        return False

def create_experiment_environment(config: ExperimentConfig):
    """创建实验环境，包括设备、服务器和任务"""
    data_generator = DataGenerator(config)
    devices = data_generator.generate_devices()
    servers = data_generator.generate_servers()
    tasks = data_generator.generate_tasks()
    
    # 为设备分配任务
    data_generator.assign_tasks_to_devices(devices, tasks)
    
    return devices, servers, tasks

def run_algorithm(algorithm_name: str, devices: List, servers: List, tasks: List, config: ExperimentConfig):
    """运行指定的算法并返回结果"""
    if algorithm_name == "local_only":
        algorithm = LocalOnlyAlgorithm(devices, servers, tasks, config)
    elif algorithm_name == "greedy":
        algorithm = GreedyAlgorithm(devices, servers, tasks, config)
    elif algorithm_name == "gnn_drl":
        algorithm = GNNDRL(devices, servers, tasks, config)
    elif algorithm_name == "gnn_il_pcr":
        algorithm = GNNILPCR(devices, servers, tasks, config)
    elif algorithm_name == "branch_and_bound":
        algorithm = BranchAndBoundAlgorithm(devices, servers, tasks, config)
    
    start_time = time.time()
    results = algorithm.execute()
    execution_time = time.time() - start_time
    
    return results, execution_time

def compare_algorithms(config: ExperimentConfig, algorithms: List[str], 
                      task_counts: List[int] = [20], num_runs: int = 3):
    """比较多个算法在不同任务数量下的性能"""
    # 结果存储 - 修改结构以支持不同任务数量和平均能耗
    results = {
        alg: {
            'task_counts': task_counts,
            'delay': [],      # 平均完成时延
            'energy': [],     # 总能耗
            'avg_energy': [], # 平均每个任务的能耗 (新增)
            'balance': [],    # 负载均衡度
            'time': []        # 算法执行时间
        } for alg in algorithms
    }
    
    # 对每个任务数量运行算法
    for task_count in task_counts:
        print(f"\n===== 任务数量: {task_count} =====")
        
        # 临时保存该任务数量下的结果
        task_results = {alg: {'delay': [], 'energy': [], 'balance': [], 'time': []} for alg in algorithms}
        
        # 为每个任务数量运行多次，取平均值以减少随机性影响
        for run in range(num_runs):
            print(f"  运行 {run+1}/{num_runs}")
            
            # 修改配置以使用特定任务数量
            temp_config = ExperimentConfig()
            temp_config.__dict__.update(config.__dict__)  # 复制原始配置
            temp_config.task_arrival_rate = task_count / 10  # 调整到达率以生成大约指定数量的任务
            
            # 为每次运行创建环境
            devices, servers, tasks = create_experiment_environment(temp_config)
            
            # 确保实际生成的任务数量与目标接近，但限制最大尝试次数
            max_attempts = 5
            attempts = 0
            while (len(tasks) < task_count * 0.8 or len(tasks) > task_count * 1.2) and attempts < max_attempts:
                devices, servers, tasks = create_experiment_environment(temp_config)
                attempts += 1
                # 每次尝试调整任务到达率
                if len(tasks) < task_count * 0.8:
                    temp_config.task_arrival_rate *= 1.2  # 增加到达率
                elif len(tasks) > task_count * 1.2:
                    temp_config.task_arrival_rate *= 0.8  # 减少到达率
            
            print(f"    实际生成任务数: {len(tasks)}")
            
            # 如果任务数为0，则跳过当前迭代
            if len(tasks) == 0 and task_count > 0:
                print(f"    警告：没有生成任务，跳过当前迭代")
                continue
            
            # 为每个算法创建独立的环境副本
            algorithm_environments = {}
            for alg in algorithms:
                # 克隆环境，以便每个算法从相同的起点开始
                devices_copy = [device.clone() for device in devices]
                servers_copy = [server.clone() for server in servers]
                algorithm_environments[alg] = (devices_copy, servers_copy)
                
            for alg in algorithms:
                print(f"    运行算法: {alg}")
                # 获取该算法的环境副本
                devices_copy, servers_copy = algorithm_environments[alg]
                
                # 运行算法
                run_results, execution_time = run_algorithm(alg, devices_copy, servers_copy, tasks, temp_config)
                delay, energy, balance = run_results
                
                # 计算平均每个任务的能耗 (新增)
                avg_energy_per_task = energy / max(1, len(tasks))
                
                # 打印服务器负载情况
                print(f"    {alg} 服务器负载情况:")
                for i, server in enumerate(servers_copy):
                    # 计算该服务器的相对负载比例而非利用率
                    load_percentage = server.current_load / server.cpu_freq * 100 if server.cpu_freq > 0 else 0
                    print(f"      服务器 {i}: 负载={server.current_load:.2e}, 容量={server.cpu_freq:.2e}, 负载比例={load_percentage:.2f}%")
                
                # 存储本次运行的结果
                task_results[alg]['delay'].append(delay)
                task_results[alg]['energy'].append(energy)
                task_results[alg]['balance'].append(balance)
                task_results[alg]['time'].append(execution_time)
                # 额外存储平均每个任务的能耗
                if 'avg_energy' not in task_results[alg]:
                    task_results[alg]['avg_energy'] = []
                task_results[alg]['avg_energy'].append(avg_energy_per_task)
        
        # 计算该任务数量下的平均值并存储
        for alg in algorithms:
            for metric in ['delay', 'energy', 'balance', 'time', 'avg_energy']:
                # 确保该指标存在
                if metric not in task_results[alg]:
                    continue
                    
                if task_results[alg][metric]:  # 确保有数据
                    avg_value = np.mean(task_results[alg][metric])
                    
                    # 确保results中有该指标的列表
                    if metric not in results[alg]:
                        results[alg][metric] = []
                        
                    results[alg][metric].append(avg_value)
                else:
                    # 如果没有数据，使用0或前一个值
                    if metric in results[alg] and results[alg][metric]:
                        results[alg][metric].append(results[alg][metric][-1])  # 使用前一个值
                    else:
                        if metric not in results[alg]:
                            results[alg][metric] = []
                        results[alg][metric].append(0)  # 使用0
            
            print(f"  {alg} 平均结果 - 完成时延: {results[alg]['delay'][-1]:.4f}, 总能耗: {results[alg]['energy'][-1]:.4e}, 平均能耗: {results[alg]['avg_energy'][-1]:.4e}, 负载均衡度: {results[alg]['balance'][-1]:.2f}%")
    
    return results

def plot_results(results: Dict, save_path: str = None, use_english: bool = True):
    """绘制比较结果图表，使用折线图展示四个算法在三个指标上的比较"""
    # 配置中文字体支持，但仍使用英文标签
    setup_chinese_font()
    
    # 强制使用英文标签
    use_english = True
    
    # 添加科学计数法格式化函数
    def sci_notation_formatter(x, pos):
        if x == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(x))))
        coef = x / 10**exp
        
        if coef == 1:
            return f'$10^{{{exp}}}$'
        else:
            return f'${coef:.1f}\\times10^{{{exp}}}$'
            
    from matplotlib.ticker import FuncFormatter
    
    algorithms = list(results.keys())
    metrics = ['delay', 'avg_energy', 'balance']  # 将energy替换为avg_energy
    
    # 确保所有算法使用相同的任务数量
    task_counts = results[algorithms[0]]['task_counts']
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 设置图表标题和y轴标签
    titles = {
        'delay': 'Average Completion Time',
        'energy': 'Total Energy Consumption',
        'avg_energy': 'Average Energy Per Task',
        'balance': 'Load Balance'
    }
    
    y_labels = {
        'delay': 'Completion Time (s)',
        'energy': 'Energy (J)',
        'avg_energy': 'Energy (J/task)',
        'balance': 'Load Balance (%)'
    }
    
    x_label = 'Number of Tasks'
    
    # 颜色映射，确保每个算法使用不同的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # 算法名称映射
    alg_names = {
        'local_only': 'Local-Only',
        'greedy': 'Greedy',
        'gnn_drl': 'GNN-DRL',
        'gnn_il_pcr': 'Proposed (GNN-IL-PCR)'
    }
    
    # 为每个指标(完成时延/平均能耗/负载均衡度)创建折线图
    for i, metric in enumerate(metrics):
        # 绘制每个算法的折线
        for j, alg in enumerate(algorithms):
            # 跳过对local_only算法负载均衡度的绘制，因为从定义上来说该值没有意义
            if metric == 'balance' and alg == 'local_only':
                continue
                
            # 使用固定的实线样式，不同的标记形状和颜色
            marker_style = markers[j % len(markers)]
            color = colors[j % len(colors)]
            
            # 获取算法显示名称
            display_name = alg_names.get(alg, alg)
            
            # 确保算法结果中包含该指标
            if metric not in results[alg]:
                print(f"警告: 算法 {alg} 没有 {metric} 指标数据")
                continue
                
            # 绘制折线 - 全部使用实线
            axes[i].plot(task_counts, results[alg][metric], 
                        marker=marker_style, markersize=8, 
                        linestyle='-', linewidth=2,
                        color=color, label=display_name)
        
        # 设置标题和坐标轴标签
        axes[i].set_title(titles[metric], fontsize=14, fontweight='bold')
        axes[i].set_xlabel(x_label, fontsize=12)
        axes[i].set_ylabel(y_labels[metric], fontsize=12)
        
        # 设置x轴刻度，确保显示所有任务数量
        axes[i].set_xticks(task_counts)
        
        # 添加网格线
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        axes[i].legend(loc='best', fontsize=10)
        
        # 根据不同指标调整y轴范围
        if metric == 'delay':
            # 设置完成时延图的y轴为线性刻度
            axes[i].set_yscale('linear')
            
            # 获取所有算法的数据
            all_values = []
            for alg in algorithms:
                if metric in results[alg] and len(results[alg][metric]) > 0:
                    all_values.extend(results[alg][metric])
            
            if all_values:
                # 找出所有值的最大值
                max_val = max(all_values)
                
                # 强制Y轴从0开始，上限设置为最大值的1.1倍
                axes[i].set_ylim(0, max_val * 1.1)
        
        elif metric == 'energy' or metric == 'avg_energy':
            # 对能耗图表使用对数刻度，以便更好地显示不同算法之间的差异
            axes[i].set_yscale('log', base=10)
            
            # 确保所有算法的数据都能在图上显示
            min_vals = []
            max_vals = []
            for alg in algorithms:
                if metric in results[alg] and len(results[alg][metric]) > 0 and any(v > 0 for v in results[alg][metric]):
                    min_vals.append(min(v for v in results[alg][metric] if v > 0))
                    max_vals.append(max(results[alg][metric]))
            
            if min_vals and max_vals:
                # 设置合适的上下限，确保所有数据都能清晰显示
                min_val = min(min_vals) * 0.8
                max_val = max(max_vals) * 1.2
                axes[i].set_ylim(min_val, max_val)
                
                # 应用自定义科学计数法格式化器
                formatter = FuncFormatter(sci_notation_formatter)
                axes[i].yaxis.set_major_formatter(formatter)
                
                # 确保负幂次能正确显示
                plt.rcParams['axes.unicode_minus'] = True
                
        elif metric == 'balance':
            # 对于负载均衡度，显示0到100%的范围
            # 找出所有算法的最大值
            max_vals = [max(results[alg][metric]) for alg in algorithms 
                       if metric in results[alg] and len(results[alg][metric]) > 0]
            
            if max_vals:  # 确保列表不为空
                max_val = max(max_vals)
                # 设置适当的上限，确保所有数据点都能显示
                upper_limit = min(100, max(20, max_val * 1.2))  # 不超过100%，至少20%
                # 确保从0开始，以便显示Local-Only的0值
                axes[i].set_ylim(0, upper_limit)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_expert_strategies(config: ExperimentConfig, save_path: str = "expert_strategies/expert_data.pkl", 
                               num_batches: int = 10, tasks_per_batch: int = 20):
    """
    生成并保存专家策略数据，作为预处理步骤
    
    参数:
        config: 实验配置
        save_path: 保存路径
        num_batches: 要生成的批次数
        tasks_per_batch: 每批次的任务数
    
    返回:
        expert_data_path: 保存的专家数据路径
    """
    import pickle
    import time
    import os
    
    # 创建保存目录
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 使用时间戳设置文件名（如果没有指定完整路径）
    if save_path.endswith('.pkl'):
        expert_data_path = save_path
    else:
        # 如果提供的是目录而不是文件路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"expert_data_{timestamp}_b{num_batches}_t{tasks_per_batch}.pkl"
        expert_data_path = os.path.join(save_path, file_name)
    
    print(f"开始生成专家策略数据...")
    print(f"配置: {num_batches}批次, 每批次{tasks_per_batch}任务")
    print(f"数据将保存到: {expert_data_path}")
    
    # 生成训练数据环境
    devices = []
    servers = []
    
    # 创建设备和服务器
    for i in range(config.num_devices):
        device = Device(
            id=i,
            cpu_freq=random.uniform(2.0, 3.0) * 1e9,  # 2-3 GHz，提升终端设备性能
            energy_coefficient=random.uniform(1.0, 2.0) * 1e-27,
            tx_power=random.uniform(0.1, 0.5),  # W
            data_rate=random.uniform(5, 20) * 1e6  # 5-20 Mbps
        )
        devices.append(device)
        
    for i in range(config.num_servers):
        server = Server(
            id=i,
            cpu_freq=random.uniform(3.0, 6.0) * 1e9,  # 3-6 GHz，降低服务器性能
            energy_coefficient=random.uniform(1.0, 2.0) * 1e-27,
            bandwidth=random.uniform(50, 100) * 1e6  # 50-100 Mbps
        )
        servers.append(server)
    
    # 收集专家数据
    expert_data = []
    data_generator = DataGenerator(config)
    
    for i in range(num_batches):
        print(f"生成专家数据批次 {i+1}/{num_batches}")
        start_time = time.time()
        
        # 生成新任务
        temp_config = ExperimentConfig()
        temp_config.__dict__.update(config.__dict__)  
        temp_config.task_arrival_rate = tasks_per_batch / 10
        
        # 生成任务并分配给设备
        new_tasks = data_generator.generate_tasks()  # 使用DataGenerator实例中已有的配置
        data_generator.assign_tasks_to_devices(devices, new_tasks)
        
        print(f"  已生成 {len(new_tasks)} 个任务")
        
        # 使用Branch and Bound算法生成专家策略
        bnb = BranchAndBoundAlgorithm(devices, servers, new_tasks, config)
        bnb.execute()
        
        # 收集专家决策数据
        batch_expert_data = bnb.get_expert_data()
        expert_data.extend(batch_expert_data)
        
        elapsed_time = time.time() - start_time
        print(f"  已收集 {len(batch_expert_data)} 条专家决策数据，耗时 {elapsed_time:.2f} 秒")
        
        # 每个批次都保存一次（增量保存）
        with open(expert_data_path, 'wb') as f:
            pickle.dump(expert_data, f)
        print(f"  已将当前数据保存到 {expert_data_path}")
    
    # 最终统计
    print(f"专家策略生成完成! 共生成 {len(expert_data)} 条专家决策数据")
    print(f"数据已保存到 {expert_data_path}")
    
    return expert_data_path

def train_and_save_models(config: ExperimentConfig, save_dir: str = "models", 
                         expert_data_path: str = None):
    """
    训练和保存算法模型
    
    参数:
        config: 实验配置
        save_dir: 模型保存目录
        expert_data_path: 专家数据文件路径，如果为None则尝试查找最新的数据文件
    """
    import pickle
    import glob
    import os
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 生成训练环境
    print("生成训练环境...")
    devices = []
    servers = []
    
    # 创建设备和服务器
    for i in range(config.num_devices):
        device = Device(
            id=i,
            cpu_freq=random.uniform(2.0, 3.0) * 1e9,  # 2-3 GHz，提升终端设备性能
            energy_coefficient=random.uniform(1.0, 2.0) * 1e-27,
            tx_power=random.uniform(0.1, 0.5),  # W
            data_rate=random.uniform(5, 20) * 1e6  # 5-20 Mbps
        )
        devices.append(device)
        
    for i in range(config.num_servers):
        server = Server(
            id=i,
            cpu_freq=random.uniform(3.0, 6.0) * 1e9,  # 3-6 GHz，降低服务器性能
            energy_coefficient=random.uniform(1.0, 2.0) * 1e-27,
            bandwidth=random.uniform(50, 100) * 1e6  # 50-100 Mbps
        )
        servers.append(server)
        
    # 生成一些测试任务
    data_generator = DataGenerator(config)
    # 生成任务并分配给设备
    tasks = data_generator.generate_tasks()
    data_generator.assign_tasks_to_devices(devices, tasks)
    
    # 初始化并训练GNN-DRL模型
    print("训练GNN-DRL模型...")
    gnn_drl = GNNDRL(devices, servers, tasks, config)
    
    # 执行一系列任务进行训练
    for _ in range(10):  # 训练10个批次
        # 生成新任务并分配给设备
        new_tasks = data_generator.generate_tasks()
        data_generator.assign_tasks_to_devices(devices, new_tasks)
        gnn_drl.tasks = new_tasks
        gnn_drl.execute()
    
    # 保存模型
    torch.save(gnn_drl.gnn.state_dict(), os.path.join(save_dir, "gnn_drl_model.pth"))
    print(f"GNN-DRL模型已保存到 {os.path.join(save_dir, 'gnn_drl_model.pth')}")
    
    # 初始化GNN-IL模型
    print("训练GNN-IL模型...")
    gnn_il = GNNILPCR(devices, servers, tasks, config)
    
    # 加载专家数据
    if expert_data_path is None:
        # 尝试查找最新的专家数据文件
        expert_files = glob.glob("expert_strategies/*.pkl")
        if not expert_files:
            print("未找到专家数据文件，无法训练GNN-IL模型")
            return
        expert_data_path = max(expert_files, key=os.path.getctime)  # 获取最新的文件
    
    print(f"从文件加载专家数据: {expert_data_path}")
    try:
        with open(expert_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        print(f"成功加载 {len(expert_data)} 条专家决策数据")
    except Exception as e:
        print(f"加载专家数据失败: {e}")
        return
    
    # 使用专家数据训练GNN-IL模型
    print("使用专家数据训练GNN-IL模型...")
    
    # 设置训练参数
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    # 初始化优化器
    optimizer = torch.optim.Adam(
        list(gnn_il.gnn.parameters()) + list(gnn_il.il_model.parameters()), 
        lr=learning_rate
    )
    
    # 损失函数
    offload_criterion = torch.nn.CrossEntropyLoss()  # 用于卸载决策（分类）
    server_criterion = torch.nn.CrossEntropyLoss()   # 用于服务器选择（分类）
    resource_criterion = torch.nn.MSELoss()          # 用于资源分配（回归）
    bandwidth_criterion = torch.nn.MSELoss()         # 用于带宽分配（回归）
    
    # 训练循环
    print("开始模型训练...")
    for epoch in range(num_epochs):
        # 打乱数据
        random.shuffle(expert_data)
        
        # 计算批次数量
        num_batches = max(1, len(expert_data) // batch_size)
        
        total_loss = 0.0
        
        for i in range(num_batches):
            # 获取当前批次数据
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(expert_data))
            batch_data = expert_data[start_idx:end_idx]
            
            # 准备批次任务
            temp_tasks = []
            for item in batch_data:
                features = item['features']
                task = Task(
                    id=features['task_id'],
                    computing_cycles=features['computing_cycles'],
                    input_data_size=features['input_data_size'],
                    output_data_size=features['output_data_size'],
                    dependencies=features['dependencies'].copy() if isinstance(features['dependencies'], list) else []
                )
                temp_tasks.append(task)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 构建批次任务图
            gnn_il.tasks = temp_tasks
            node_features, edge_index = gnn_il._build_task_graph()
            
            # 前向传播
            task_embeddings = gnn_il.gnn(node_features, edge_index)
            
            # 计算批次损失
            batch_loss = 0.0
            
            for j, item in enumerate(batch_data):
                # 获取任务嵌入
                task_emb = task_embeddings[j].unsqueeze(0)
                
                # 前向传播
                offload_pred, server_pred, resource_pred, bandwidth_pred = gnn_il.il_model(task_emb)
                
                # 准备标签
                offload_label = torch.tensor([item['label']['offload']], dtype=torch.long)
                server_id = max(0, item['label']['server_id'])  # 确保非负
                server_label = torch.tensor([server_id], dtype=torch.long)
                resource_label = torch.tensor([[item['label']['resource_allocation']]], dtype=torch.float)
                bandwidth_label = torch.tensor([[item['label']['bandwidth_allocation']]], dtype=torch.float)
                
                # 计算损失
                offload_loss = offload_criterion(offload_pred, offload_label)
                server_loss = server_criterion(server_pred, server_label) if item['label']['offload'] == 1 else 0
                resource_loss = resource_criterion(resource_pred, resource_label) if item['label']['offload'] == 1 else 0
                bandwidth_loss = bandwidth_criterion(bandwidth_pred, bandwidth_label) if item['label']['offload'] == 1 else 0
                
                # 计算总损失
                task_loss = offload_loss
                if item['label']['offload'] == 1:
                    task_loss += server_loss + resource_loss + bandwidth_loss
                
                batch_loss += task_loss
            
            # 计算平均批次损失
            if len(batch_data) > 0:
                batch_loss = batch_loss / len(batch_data)
                
                # 反向传播和优化
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 每10个周期保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(gnn_il.gnn.state_dict(), os.path.join(save_dir, f"gnn_il_model_epoch{epoch+1}.pth"))
            torch.save(gnn_il.il_model.state_dict(), os.path.join(save_dir, f"il_model_epoch{epoch+1}.pth"))
    
    torch.save(gnn_il.gnn.state_dict(), os.path.join(save_dir, "gnn_il_model.pth"))
    torch.save(gnn_il.il_model.state_dict(), os.path.join(save_dir, "il_model.pth"))
    
    print(f"模型已保存到 {save_dir} 目录")

def main():
    # 设置中文字体支持
    has_chinese_font = setup_chinese_font()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="边缘计算任务卸载算法评估")
    parser.add_argument("--mode", type=str, default="compare", 
                        choices=["preprocess", "train", "compare", "single"],
                        help="运行模式: preprocess (生成专家策略数据), train (训练模型), compare (比较算法), single (运行单个算法)")
    parser.add_argument("--algorithm", type=str, default="gnn_il_pcr",
                        choices=["local_only", "greedy", "gnn_drl", "gnn_il_pcr", "branch_and_bound"],
                        help="要运行的算法")
    parser.add_argument("--num_runs", type=int, default=5, help="每个任务数量下的运行次数")
    parser.add_argument("--task_min", type=int, default=10, help="最小任务数量")
    parser.add_argument("--task_max", type=int, default=100, help="最大任务数量")
    parser.add_argument("--task_step", type=int, default=10, help="任务数量步长")
    parser.add_argument("--num_devices", type=int, default=50, help="终端设备数量")
    parser.add_argument("--num_servers", type=int, default=10, help="边缘服务器数量")
    parser.add_argument("--save_plot", type=str, default="results/comparison.png", help="保存图表的路径")
    parser.add_argument("--use_english", action="store_true", help="使用英文显示图表标题和标签")
    
    # 新增参数 - 用于预处理和训练模式
    parser.add_argument("--expert_data", type=str, default=None, 
                        help="专家策略数据文件路径 (用于训练模式)")
    parser.add_argument("--expert_save_path", type=str, default="expert_strategies/expert_data.pkl", 
                        help="专家策略数据保存路径 (用于预处理模式)")
    parser.add_argument("--num_batches", type=int, default=10, 
                        help="生成专家策略的批次数 (用于预处理模式)")
    parser.add_argument("--tasks_per_batch", type=int, default=20, 
                        help="每批次的任务数量 (用于预处理模式)")
    
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig()
    
    # 根据命令行参数调整配置
    config.num_devices = args.num_devices
    config.num_servers = args.num_servers
    
    # 运行相应的模式
    if args.mode == "preprocess":
        print("===== 预处理模式: 生成专家策略数据 =====")
        # 确保保存目录存在
        os.makedirs(os.path.dirname(args.expert_save_path), exist_ok=True)
        # 生成专家策略数据
        expert_data_path = generate_expert_strategies(
            config, 
            save_path=args.expert_save_path,
            num_batches=args.num_batches,
            tasks_per_batch=args.tasks_per_batch
        )
        print(f"专家策略数据生成完成，已保存至: {expert_data_path}")
        
    elif args.mode == "train":
        print("===== 训练模式: 训练和保存模型 =====")
        # 使用预生成的专家数据文件训练模型
        train_and_save_models(config, expert_data_path=args.expert_data)
        
    elif args.mode == "compare":
        print("===== 比较模式: 比较算法性能 =====")
        # 生成任务数量序列
        task_counts = list(range(args.task_min, args.task_max + 1, args.task_step))
        
        # 调用比较函数
        # 注意：Branch and Bound算法不应作为在线算法参与比较
        # 它的设计用途是在离线阶段生成专家策略数据(在preprocess模式中使用)
        algorithms = ["local_only", "greedy", "gnn_drl", "gnn_il_pcr"]
        results = compare_algorithms(config, algorithms, task_counts, args.num_runs)
        
        # 绘制结果
        plot_results(results, args.save_plot, use_english=args.use_english)
        
        # 保存结果到文件
        save_dir = os.path.dirname(args.save_plot)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        results_file = os.path.join(save_dir, "comparison_results.json")
        try:
            with open(results_file, 'w') as f:
                # 将numpy数组转换为列表以便JSON序列化
                serializable_results = {}
                for alg in results:
                    serializable_results[alg] = {}
                    for metric in results[alg]:
                        if isinstance(results[alg][metric], np.ndarray):
                            serializable_results[alg][metric] = results[alg][metric].tolist()
                        else:
                            serializable_results[alg][metric] = results[alg][metric]
                
                json.dump(serializable_results, f, indent=2)
            print(f"结果已保存至 {results_file}")
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    elif args.mode == "single":
        print(f"===== 单算法模式: 运行 {args.algorithm} =====")
        # 创建环境
        devices, servers, tasks = create_experiment_environment(config)
        
        # 运行单个算法
        results, execution_time = run_algorithm(args.algorithm, devices, servers, tasks, config)
        delay, energy, balance = results
        
        print("\n===== 结果 =====")
        print(f"平均完成时延: {delay:.4f}秒")
        print(f"设备能耗: {energy:.4e}焦耳")
        print(f"负载均衡度: {balance:.2f}%")
        print(f"执行时间: {execution_time:.4f}秒")
    
    print("程序执行完成")

if __name__ == "__main__":
    main()
