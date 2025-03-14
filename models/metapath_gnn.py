import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer

class MetaPathGNN(nn.Module):
    def __init__(self, user_dim, item_dim, factor_dim, hidden_dim, num_heads=4):
        super(MetaPathGNN, self).__init__()
        
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.factor_dim = factor_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 노드 타입별 임베딩 투영: 입력 차원 -> hidden_dim
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.item_proj = nn.Linear(item_dim, hidden_dim)
        self.user_factor_proj = nn.Linear(factor_dim, hidden_dim)
        self.item_factor_proj = nn.Linear(factor_dim, hidden_dim)
        
        # 의미적 메타패스 GAT 레이어: 출력 차원은 hidden_dim * num_heads
        self.gat_u_q_u = GATConv(hidden_dim, hidden_dim, heads=num_heads)
        self.gat_i_q_i = GATConv(hidden_dim, hidden_dim, heads=num_heads)
        
        # 상호작용 메타패스 GAT 레이어
        # 여기서 in_channels를 hidden_dim * num_heads (예: 256)로 맞춰줍니다.
        self.gat_u_i = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
        self.gat_i_u = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads)
    
    def forward(self, node_features, metapath_edge_indices):
        # 노드 임베딩 투영
        user_emb = self.user_proj(node_features['user'])
        item_emb = self.item_proj(node_features['item'])
        user_factor_embs = [self.user_factor_proj(level_emb) for level_emb in node_features['user_factor']]
        item_factor_embs = [self.item_factor_proj(level_emb) for level_emb in node_features['item_factor']]
        
        # 모든 임베딩을 row-wise로 결합
        all_emb = torch.cat([user_emb, item_emb] + user_factor_embs + item_factor_embs, dim=0)
        
        # 첫 번째 GAT 계층: concat=True이므로 출력 차원은 hidden_dim*num_heads
        u_q_u_emb = self.gat_u_q_u(all_emb, metapath_edge_indices['u_q_u'])
        i_q_i_emb = self.gat_i_q_i(all_emb, metapath_edge_indices['i_q_i'])
        
        num_users = user_emb.size(0)
        num_items = item_emb.size(0)
        
        H_u = u_q_u_emb[:num_users]
        H_i = i_q_i_emb[num_users:num_users + num_items]
        
        combined_emb = torch.cat([H_u, H_i], dim=0)
        
        # 두 번째 GAT 계층: 이제 in_channels는 hidden_dim*num_heads (256)로 맞춰짐
        H_hat_u = self.gat_i_u(combined_emb, metapath_edge_indices['u_i'])[:num_users]
        H_hat_i = self.gat_u_i(combined_emb, metapath_edge_indices['i_u'])[num_users: num_users + num_items]
        
        return H_hat_u, H_hat_i
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        