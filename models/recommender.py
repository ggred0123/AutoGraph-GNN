import torch
import torch.nn as nn

class AutoGraphRecommender(nn.Module):
    """AutoGraph 추천 모델"""
    def __init__(self, user_dim, item_dim, hidden_dim, output_dim, num_heads=4):
        super(AutoGraphRecommender, self).__init__()
        
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 사용자, 아이템 임베딩을 hidden_dim 차원으로 매핑
        self.user_encoder = nn.Linear(user_dim, hidden_dim)
        self.item_encoder = nn.Linear(item_dim, hidden_dim)
        
        # 최종 concatenated vector 차원: 
        # user_emb (hidden_dim) + item_emb (hidden_dim) + 
        # user_graph_emb (hidden_dim * num_heads) + item_graph_emb (hidden_dim * num_heads)
        total_dim = hidden_dim * 2 + hidden_dim * num_heads * 2
        
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
    
    def forward(self, user_features, item_features, user_graph_emb, item_graph_emb):
        user_emb = self.user_encoder(user_features)
        item_emb = self.item_encoder(item_features)
        
        # Concatenate along feature dimension
        combined = torch.cat([user_emb, item_emb, user_graph_emb, item_graph_emb], dim=1)
        score = self.combiner(combined)
        
        return score
