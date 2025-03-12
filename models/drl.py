import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DRLAgent:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.4  # 提高初始探索率，从0.2增加到0.4
        self.epsilon_min = 0.1  # 提高最小探索率，从0.05增加到0.1
        self.epsilon_decay = 0.99  # 保持探索率衰减速度不变
        self.gamma = 0.99   # 折扣因子
        self.test_mode = False  # 测试模式标志，在测试时禁用探索
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)
        
        # 负载均衡相关数据
        self.server_load_history = {}  # 记录服务器负载历史
        
    def select_action(self, state: torch.Tensor, server_loads: dict = None) -> int:
        """
        选择动作，同时考虑服务器负载情况
        
        参数:
            state: 当前状态
            server_loads: 服务器当前负载字典 {server_id: load_percentage}
        """
        # 确定是否进行随机探索
        # 在测试模式下使用更高的探索率(0.1)
        should_explore = np.random.random() < (self.epsilon if not self.test_mode else 0.1)
        
        if should_explore:
            # 随机探索情况下，偏向选择低负载的服务器
            if server_loads is not None:
                # 根据负载程度计算选择概率
                server_probs = []
                total_servers = len(server_loads)
                
                # 对于动作0(本地执行)和无效服务器ID，使用默认概率
                server_probs.append(1.0 / (total_servers + 1))  # 动作0的概率
                
                # 计算其他服务器的选择概率
                load_sum = sum(load for load in server_loads.values())
                if load_sum > 0:
                    for server_id in range(total_servers):
                        # 负载越低，概率越高
                        load = server_loads.get(server_id, 0)
                        inverse_load = 1.0 - (load / max(load_sum, 1e-6))
                        server_probs.append(inverse_load * 2.0 / (total_servers + 1))
                else:
                    # 如果所有服务器负载为0，均匀分配概率
                    for _ in range(total_servers):
                        server_probs.append(1.0 / (total_servers + 1))
                
                # 归一化概率
                prob_sum = sum(server_probs)
                server_probs = [p / prob_sum for p in server_probs]
                
                # 根据概率选择动作
                actions = list(range(self.action_dim))
                return np.random.choice(actions, p=server_probs)
            else:
                # 无服务器负载信息时进行普通随机探索
                return np.random.randint(self.action_dim)
        
        # 根据Q值选择最佳动作
        with torch.no_grad():
            q_values = self.q_network(state)
            
            # 应用负载均衡偏好
            if server_loads is not None:
                # 为了负载均衡，对高负载服务器的Q值进行惩罚
                q_values_np = q_values.numpy() if isinstance(q_values, torch.Tensor) else q_values
                for server_id, load in server_loads.items():
                    if server_id + 1 < len(q_values_np):  # +1是因为动作0表示本地执行
                        # 负载越高，惩罚越大
                        q_values_np[server_id + 1] -= load * 0.2  # 添加负载惩罚因子
                
                # 转回tensor
                q_values = torch.tensor(q_values_np, dtype=q_values.dtype)
            
            return torch.argmax(q_values).item()
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def set_test_mode(self, test_mode: bool = True):
        """设置测试模式，在测试时将探索率设为较低值而不是完全禁用"""
        self.test_mode = test_mode
        if test_mode:
            # 测试模式下保留一定的探索率(0.1)，提高负载均衡能力
            self.epsilon_backup = self.epsilon  # 备份当前探索率
            self.epsilon = 0.1  # 提高测试模式探索率为原来(0.05)的2倍
            print(f"DRL代理已切换到测试模式，探索率设为{self.epsilon:.4f}")
        else:
            # 恢复原来的探索率
            if hasattr(self, 'epsilon_backup'):
                self.epsilon = self.epsilon_backup
    
    def update(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done).float()  # 确保是浮点数类型
        
        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.soft_update()
        
    def soft_update(self, tau: float = 0.01):
        """软更新目标网络"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, 
             next_state: torch.Tensor, done: torch.Tensor):
        """
        添加经验到回放记忆
        参数:
            state: 当前状态，形状为 [batch_size, state_dim]
            action: 执行的动作，形状为 [batch_size, 1]
            reward: 获得的奖励，形状为 [batch_size, 1]
            next_state: 下一个状态，形状为 [batch_size, state_dim]
            done: 是否结束，形状为 [batch_size, 1]
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)
        
    def __len__(self) -> int:
        return len(self.memory)