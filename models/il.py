import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ImitationLearning(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多头输出
        self.offload_head = nn.Linear(hidden_dim, 2)     # 卸载决策
        self.server_head = nn.Linear(hidden_dim, 1)      # 服务器选择
        self.resource_head = nn.Linear(hidden_dim, 2)    # 资源分配
        
        self.fisher_information = {}  # 用于EWC
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 检查输入形状
        print(f"IL模型forward - 输入形状: {x.shape}")
        
        # 确保输入是二维的 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加批次维度
        
        # 打印调整后的形状
        print(f"IL模型forward - 调整后形状: {x.shape}")
        
        features = self.network(x)
        return {
            'offload': torch.sigmoid(self.offload_head(features)),
            'server': self.server_head(features),
            'resource': self.resource_head(features)
        }
    
    def calculate_fisher_matrix(self, data_loader: torch.utils.data.DataLoader):
        """计算Fisher信息矩阵"""
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        
        self.eval()
        for batch in data_loader:
            self.zero_grad()
            output = self(batch['input'])
            loss = F.mse_loss(output['offload'], batch['target'])
            loss.backward()
            
            for n, p in self.named_parameters():
                fisher_dict[n] += p.grad.data ** 2 / len(data_loader)
        
        self.fisher_information = fisher_dict
        
    def update_with_ewc(self, new_data: torch.utils.data.DataLoader, 
                       old_params: Dict[str, torch.Tensor], lambda_ewc: float = 1000):
        """使用EWC进行在线更新"""
        for batch in new_data:
            self.train()
            self.zero_grad()
            
            # 计算当前任务的损失
            output = self(batch['input'])
            loss = F.mse_loss(output['offload'], batch['target'])
            
            # 添加EWC惩罚项
            for n, p in self.named_parameters():
                loss += (lambda_ewc / 2) * torch.sum(
                    self.fisher_information[n] * (p - old_params[n]) ** 2
                )
            
            loss.backward()
            self.optimizer.step()