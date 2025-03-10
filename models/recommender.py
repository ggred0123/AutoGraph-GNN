import torch
import torch.nn as nn


class AutoGraphRecommender(nn.Module):
    """AutoGraph 추천 모델"""
    def __init__(self, user_dim, item_dim, hidden_dim, output_dim):
        super(AutoGraphRecommender, self).__init__()
        
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        
        
        self.user_encoder = nn.Linear(user_dim, hidden_dim)
        self.item_encoder = nn.Linear(item_dim, hidden_dim)
        
        
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
    def forward(self, user_features, item_features, user_graph_emb, item_graph_emb):
        """
        순방향 전파 함수
        
        Args:
            user_features: 사용자 특성 텐서
            item_features: 아이템 특성 텐서
            user_graph_emb: 사용자 그래프 임베딩 텐서
            item_graph_emb: 아이템 그래프 임베딩 텐서
            
        Returns:
            사용자 아이템 선호도 점수
        """
        
        user_emb = self.user_encoder(user_features)
        item_emb = self.item_encoder(item_features)
        
        combined = torch.cat([user_emb, item_emb, user_graph_emb, item_graph_emb], dim=1)
        score = self.combiner(combined)
        
        return score
        
        
        
        
        
        
        