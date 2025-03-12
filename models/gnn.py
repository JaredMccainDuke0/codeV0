import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TaskGNN(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=n_heads)
        self.conv2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1)
        
    def forward(self, x, edge_index):
        """
        参数:
            x: 节点特征矩阵
            edge_index: 边的连接关系
        返回:
            node_embeddings: 节点嵌入
        """
        # 检查是否为空图（无节点或无边）
        if x.size(0) == 0 or isinstance(edge_index, torch.Tensor) and edge_index.numel() == 0:
            # 返回空的嵌入张量
            return torch.tensor([])
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)

    def get_attention_weights(self, x, edge_index):
        """获取注意力权重"""
        _, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        return attention_weights