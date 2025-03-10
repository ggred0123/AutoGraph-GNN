import torch

class AutoGraphConstructor:
    """
    자동 그래프 생성 모델
    """
    def __init__(self,user_vq, item_vq):
        
        """
        생성자
        Args:
            user_vq (ResidualVectorQuantizer): 사용자 벡터 양자화 모델
            item_vq (ResidualVectorQuantizer): 아이템 벡터 양자화 모델
        """
        self.user_vq = user_vq
        self.item_vq = item_vq
        
    def construct_graph(self, user_vectors, item_vectors, interactions):
        """
        그래프 생성 함수
        Args:
            user_vectors (torch.Tensor): 사용자 벡터 텐서
            item_vectors (torch.Tensor): 아이템 벡터 텐서
            interactions (torch.Tensor): 사용자 - 아이템 상호작용 정보 ( user_id, item_id) 쌍의 리스트
            
        Returns:
            노드 특성, 에지 인덱스( 유형별)
        """
        
        user_quantized, user_indices, _ , _ = self.user_vq(user_vectors) # 사용자 벡터 양자화
        item_quantized, item_indices, _ , _ = self.item_vq(item_vectors) # 아이템 벡터 양자화
        
        node_features = {} # 노드 특성 저장
        node_features['user'] = user_vectors # 사용자 벡터 저장
        node_features['item'] = item_vectors # 아이템 벡터 저장
        
        user_factor_embeddings = [cb.clone() for cb in self.user_vq.codebooks] # 사용자 인코더 코드북 복사
        item_factor_embeddings = [cb.clone() for cb in self.item_vq.codebooks] # 아이템 인코더 코드북 복사
        
        node_features['user_factor'] = user_factor_embeddings # 사용자 인코더 코드북 저장
        node_features['item_factor'] = item_factor_embeddings # 아이템 인코더 코드북 저장
        
        edge_index = {} # 에지 인덱스 저장
        
        user_item_edges = torch.tensor(interactions).t() # 사용자 - 아이템 상호작용 정보
        edge_index['user_item'] = user_item_edges
        
        user_factor_edges = [] # 사용자 인코더 코드북 에지 저장
        for user_id , user_factor_ids in enumerate(zip(*user_indices)): # 사용자 인덱스 순회
            for level, factor_id in enumerate(user_factor_ids): # 인코더 코드북 순회
                user_factor_edges.append([user_id, factor_id.item() +level * self.user_vq.codebook_size]) # 사용자 인덱스와 코드북 인덱스 결합
                
        edge_index['user_factor'] = torch.tensor(user_factor_edges).t() # 사용자 인코더 코드북 에지 저장
        
        item_factor_edges = [] # 아이템 인코더 코드북 에지 저장
        for item_id, item_factor_ids in enumerate(zip(*item_indices)): # 아이템 인덱스 순회
            for level, factor_id in enumerate(item_factor_ids): # 인코더 코드북 순회
                item_factor_edges.append([item_id, factor_id.item() +level * self.item_vq.codebook_size]) # 아이템 인덱스와 코드북 인덱스 결합
                
        edge_index['item_factor'] = torch.tensor(item_factor_edges).t() # 아이템 인코더 코드북 에지 저장
        
        return node_features, edge_index
        
        
        
        
        
        
        
            
        