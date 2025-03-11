import torch
from models.semantic_generator import SemanticVectorGenerator
from models.vector_quantizer import ResidualVectorQuantizer
from models.graph_constructor import AutoGraphConstructor
from models.recommender import AutoGraphRecommender
from models.metapath_gnn import MetaPathGNN
from utils.visualization import visualize_residual_quantization, visualize_graph_construction


class AutoGraphPipeline:
    
    """AutoGraph 전체 파이프라인"""
    
    
    def __init__(self, config):
        """
        파이프라인 초기화
        """
        
        self.config = config
        
        #의미 벡터 생성기 초기화
        self.semantic_generator = SemanticVectorGenerator( 
            llm_model_name = config.llm_model_name
        )
        
        # 잔차 양자화 모델 초기화
        self.user_vq = ResidualVectorQuantizer( 
            input_dim = config.vector_dim,
            hidden_dim = config.hidden_dim,
            codebook_size = config.codebook_size,
            num_codebooks = config.num_codebooks,            
        )
        
        # 잔차 양자화 모델 초기화
        self.item_vq = ResidualVectorQuantizer(
            input_dim = config.vector_dim,
            hidden_dim = config.hidden_dim,
            codebook_size = config.codebook_size,
            num_codebooks = config.num_codebooks,
        )
        
        # 그래프 생성기 초기화
        self.graph_constructor = AutoGraphConstructor(
            user_vq = self.user_vq,
            item_vq = self.item_vq
        )
        
        # 메타패스 GNN 초기화
        self.metapath_gnn = MetaPathGNN(
            user_dim = config.sementic_vector_dim,
            item_dim = config.sementic_vector_dim,
            factor_dim = config.hidden_dim,
            hidden_dim = config.hidden_dim,
            num_heads = config.num_heads,
        )
        
        # 추천 모델 초기화
        self.recommender = AutoGraphRecommender(
            user_dim = config.sementic_vector_dim,
            item_dim = config.sementic_vector_dim,
            hidden_dim = config.hidden_dim,
            output_dim = config.output_dim
        )
    
    # 의미 벡터 생성 단계
    def generate_semantic_vectors(self, user_profiles, user_histories, item_attributes):
        """
        의미 벡터 생성
        """
        
        user_prompts = self.semantic_generator.generate_user_prompts(user_profile, user_histories)
        
        item_prompts = self.semantic_generator.generate_item_prompts(item_attributes)
        
        user_vectors = self.semantic_generator.generate_user_embeddings(
            user_profiles, self.interactions_df, item_attributes, 
            method='weighted', batch_size=16
        )
        item_vectors = self.semantic_generator.generate_book_embeddings(
        item_attributes, batch_size=16, combine_method='concat'
    )
        
        return user_vectors, item_vectors
    
    # 잠재 요인 추출 단계
    def extract_latent_factors(self, user_vectors, item_vectors):
        """잠재 요인 추출 단계""" 
        
        user_quantized, user_indices, _ , user_loss = self.user_vq(user_vectors)
        
        item_quantized, item_indices, _ , item_loss = self.item_vq(item_vectors)
        
        return user_quantized, item_quantized, user_indices, item_indices
    
    # 그래프 구성 단계
    def construct_graph(self, user_vectors, item_vectors, interactions):
        """그래프 구성 단계"""
        
        node_features, edge_indices = self.graph_constructor.construct_graph(user_vectors, item_vectors, interactions)
        
        return node_features, edge_indices
    
    # 메타패스 에지 준비 단계
    def prepare_metapath_edges(self, edge_indices):
        """메타패스 에지 준비 단계"""
        
        metapath_edge_indices = {}
        
        metapath_edge_indices['u_q_u'] = edge_indices['user_factor'] # 임시처리
        
        metapath_edge_indices['i_q_i'] = edge_indices['item_factor'] # 임시처리
        
        metapath_edge_indices['u_i_u'] = edge_indices['user_item'] # 임시처리
        
        metapath_edge_indices['i_u_i'] = edge_indices['item_user'] # 임시처리
        
        return metapath_edge_indices
    
    # 메시지 전파 단계
    def perform_message_propagation(self, node_features, metapath_edge_indices):
        """메시지 전파 단계"""
        
        user_graph_emb, item_graph_emb = self.metapath_gnn(node_features, metapath_edge_indices)
        
        return user_graph_emb, item_graph_emb
    
    # 추천 단계
    def run(self,dataset):
        """전체 파이프라인 실행
        
        Args: dataset: 데이터셋 객체"""
        
        user_profiles = dataset.get_user_profiles()
        
        user_histories = dataset.get_user_histories()
        
        item_attributes = dataset.get_item_attributes()
        
        interactions = dataset.get_interactions()
        
        user_vectors, item_vectors = self.generate_semantic_vectors(user_profiles, user_histories, item_attributes)
        
        user_quantized, item_quantized, user_indices, item_indices = self.extract_latent_factors(user_vectors, item_vectors)
        
        node_features, edge_indices = self.construct_graph(user_vectors, item_vectors, interactions)
        
        metapath_edge_indices = self.prepare_metapath_edges(edge_indices)
        
        user_graph_emb, item_graph_emb = self.perform_message_propagation(node_features, metapath_edge_indices)
        
        results = {
            'user_vectors': user_vectors,
            'item_vectors': item_vectors,
            'user_quantized': user_quantized,
            'item_quantized': item_quantized,
            'user_indices': user_indices,
            'item_indices': item_indices,
            'node_features': node_features,
            'edge_indices': edge_indices,
            'user_graph_emb': user_graph_emb,
            'item_graph_emb': item_graph_emb,
            'models': {
                'semantic_generator': self.semantic_generator,
                'user_vq': self.user_vq,
                'item_vq': self.item_vq,
                'graph_constructor': self.graph_constructor,
                'metapath_gnn': self.metapath_gnn,
                'recommender': self.recommender
            },
            
        }
        
        return results
        
  