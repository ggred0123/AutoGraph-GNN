import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
import seaborn as sns


def visualize_residual_quantization(input_dim = 128, hidden_dim = 32, codebook_size = 10, num_codebooks = 3, num_samples = 100):
    """
    잔차 양자화 모델의 시각화
    
    Args:
        input_dims: 입력 벡터 차원
        hidden_dim: 은닉층 벡터 차원
        num_codebooks: 양자화 코드북 수
        num_samples: 시각화할 샘플 수
        codebook_size: 각 코드북의 벡터 수
        
    
    """
    
    
    np.random.seed(42)
    input_vectors = torch.randn(num_samples, input_dim)
    
    hidden_vectors = torch.FloatTensor(np.random.normal(0,1, (num_samples, hidden_dim)))
    
    codebooks = []
    
    for i in range(num_codebooks):
        
        scale = 1.0 / (i+1)
        codebook = torch.FloatTensor(np.random.normal(0,scale, (codebook_size, hidden_dim)))
        codebooks.append(codebook)
    
    residual = hidden_vectors.clone()
    quantized_vectors = []
    indices_per_level = []
    residuals_per_level = [residual.clone()]
    
    for i in range(num_codebooks):
        codebook = codebooks[i]
        
        distances = torch.cdist(residual, codebook)
        
        min_indices = torch.argmin(distances, dim =1)
        indices_per_level.append(min_indices)
        
        selected_vectors = codebook[min_indices]
        quantized_vectors.append(selected_vectors)
        
        residual = residual - selected_vectors
        residuals_per_level.append(residual.clone())
    
    pca = PCA(n_components=2)
    
    hidden_vectors_2d = pca.fit_transform(hidden_vectors.numpy())
    
    
    codebooks_2d = []
    for codebook in codebooks:
        codebooks_2d.append(pca.transform(codebook.numpy()))
    
    plt.subplot(2,2,1)
    plt.scatter(hidden_vectors_2d[:,0], hidden_vectors_2d[:,1], alpha=0.6, label='Hidden Vectors')
    
    colors = sns.color_palette('husl', num_codebooks)
    
    for i, codebook_2d in enumerate(codebooks_2d):
        plt.scatter(codebook_2d[:,0], codebook_2d[:,1],s=100, color=colors[i],marker= "*", label=f'Codebook Level{i+1}', alpha=0.8)
        
    plt.title('Hidden Vectors and Codebooks')
    plt.legend()
    plt.axis('equal')
   
    plt.subplot(2, 2, 2)
    
    for i, residual in enumerate(residuals_per_level):
        residual_2d = pca.transform(residual.numpy())
        if i == 0:
            label = 'Original Vectors'
        else:
            label = f'After Level {i}'
        
        plt.scatter(residual_2d[:, 0], residual_2d[:, 1], alpha=0.6, label=label)
    
    plt.title('Residual Vectors after Each Quantization Level')
    plt.legend()
    plt.axis('equal')
    
    # 3. 샘플 벡터의 양자화 시각화
    plt.subplot(2, 2, 3)
    
    # 5개 샘플만 선택하여 시각화
    sample_indices = np.random.choice(num_samples, 5, replace=False)
    
    for idx in sample_indices:
        # 원본 벡터
        x, y = hidden_vectors_2d[idx]
        plt.scatter(x, y, s=150, alpha=0.5, label=f'Sample {idx} Original' if idx == sample_indices[0] else "")
        
        # 현재 포인트 위치
        current_point = np.array([x, y])
        
        # 각 레벨에서의 양자화 진행 표시
        for level in range(num_codebooks):
            codebook_idx = indices_per_level[level][idx].item()
            vec = codebooks_2d[level][codebook_idx]
            
            # 화살표로 양자화 방향 표시
            plt.arrow(current_point[0], current_point[1], 
                     vec[0] - current_point[0], vec[1] - current_point[1], 
                     head_width=0.05, head_length=0.1, fc=colors[level], ec=colors[level], alpha=0.6)
            
            # 선택된 코드북 벡터 하이라이트
            plt.scatter(vec[0], vec[1], s=150, color=colors[level], alpha=0.8, 
                       marker='*', edgecolors='black')
            
            # 다음 포인트 업데이트
            current_point = vec
    
    plt.title('Vector Quantization Process (5 Samples)')
    plt.axis('equal')
    
    # 4. 각 레벨의 코드북 활성화 비율 시각화
    plt.subplot(2, 2, 4)
    
    activation_counts = []
    for level_indices in indices_per_level:
        # 각 코드북 벡터가 몇 번 선택되었는지 계산
        counts = np.zeros(codebook_size)
        for idx in level_indices:
            counts[idx] += 1
        activation_counts.append(counts)
    
    x = np.arange(codebook_size)
    bar_width = 0.2
    
    for i, counts in enumerate(activation_counts):
        plt.bar(x + i*bar_width, counts/num_samples, 
               width=bar_width, color=colors[i], label=f'Level {i+1}')
    
    plt.xlabel('Codebook Vector Index')
    plt.ylabel('Activation Ratio')
    plt.title('Codebook Vectors Activation Ratio')
    plt.xticks(x + bar_width, [str(i) for i in range(codebook_size)])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('residual_quantization_visualization.png', dpi=300)
    plt.show()
    
    return {
        'hidden_vectors': hidden_vectors,
        'codebooks': codebooks,
        'indices_per_level': indices_per_level,
        'residuals_per_level': residuals_per_level
    }
def visualize_graph_construction(hidden_vectors, codebooks, indices_per_level, num_vis_samples=30):
    """
    잔차 양자화가 그래프 구조에 미치는 영향을 시각화합니다.
    
    Args:
        hidden_vectors: 히든 벡터 텐서
        codebooks: 코드북 리스트
        indices_per_level: 각 레벨의 인덱스
        num_vis_samples: 시각화할 샘플 수
    """
    num_samples = hidden_vectors.shape[0]
    num_codebooks = len(codebooks)
    
    # 시각화를 위해 샘플 수 제한
    if num_samples > num_vis_samples:
        sample_indices = np.random.choice(num_samples, num_vis_samples, replace=False)
        hidden_vectors = hidden_vectors[sample_indices]
        indices_per_level = [indices[sample_indices] for indices in indices_per_level]
    else:
        sample_indices = np.arange(num_samples)
    
    # 2D로 차원 축소
    pca = PCA(n_components=2)
    hidden_vectors_2d = pca.fit_transform(hidden_vectors.numpy())
    
    # 그래프 구조 시각화
    plt.figure(figsize=(18, 15))
    
    # 1. 원본 그래프 (잠재 요인 없음)
    plt.subplot(2, 2, 1)
    plt.scatter(hidden_vectors_2d[:, 0], hidden_vectors_2d[:, 1], s=100, alpha=0.7, c='blue', label='Items')
    
    # k-nearest neighbors로 아이템 간 관계 형성 (간단한 시뮬레이션)
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(hidden_vectors_2d)
    distances, indices = nn.kneighbors(hidden_vectors_2d)
    
    # 이웃 관계 시각화
    for i in range(len(hidden_vectors_2d)):
        for j in indices[i][1:]:  # 자기 자신 제외
            plt.plot([hidden_vectors_2d[i, 0], hidden_vectors_2d[j, 0]], 
                    [hidden_vectors_2d[i, 1], hidden_vectors_2d[j, 1]], 
                    'k-', alpha=0.2)
    
    plt.title('Original Graph (Without Latent Factors)')
    plt.legend()
    
    # 2-4. 각 레벨의 잠재 요인을 추가한 그래프
    colors = sns.color_palette("husl", num_codebooks)
    
    for level in range(min(3, num_codebooks)):
        plt.subplot(2, 2, level+2)
        
        # 아이템 노드 표시
        plt.scatter(hidden_vectors_2d[:, 0], hidden_vectors_2d[:, 1], s=100, alpha=0.7, c='blue', label='Items')
        
        # 선택된 잠재 요인 노드 표시
        selected_factors = set()
        for i in range(len(hidden_vectors_2d)):
            factor_idx = indices_per_level[level][i].item()
            selected_factors.add(factor_idx)
        
        # 잠재 요인 위치 계산 (임의로 배치)
        np.random.seed(42 + level)
        factor_pos = np.random.normal(0, 3, (len(selected_factors), 2))
        
        # 선택된 잠재 요인 표시
        factor_idx_to_pos = {}
        for i, factor_idx in enumerate(selected_factors):
            x, y = factor_pos[i]
            plt.scatter(x, y, s=200, color=colors[level], marker='*', 
                       edgecolors='black', alpha=0.9, label='Latent Factor' if i == 0 else "")
            factor_idx_to_pos[factor_idx] = (x, y)
        
        # 아이템과 선택된 잠재 요인 간 연결
        for i in range(len(hidden_vectors_2d)):
            factor_idx = indices_per_level[level][i].item()
            if factor_idx in factor_idx_to_pos:
                fx, fy = factor_idx_to_pos[factor_idx]
                plt.plot([hidden_vectors_2d[i, 0], fx], 
                        [hidden_vectors_2d[i, 1], fy], 
                        '-', color=colors[level], alpha=0.4)
        
        plt.title(f'Graph with Level {level+1} Latent Factors')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('graph_construction_visualization.png', dpi=300)
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt

def visualize_user_item_graph(node_features, edge_indices, users_df, books_df, filename="user_item_graph.png"):
    """
    사용자-책 그래프를 시각화하는 함수입니다.
    
    Args:
        node_features (dict): 노드 특성 딕셔너리 (여기서는 사용하지 않음)
        edge_indices (dict): 에지 인덱스 딕셔너리, 'user_item' 키를 사용
        users_df (pd.DataFrame): 사용자 정보 DataFrame (컬럼 'user_id' 포함, 1-indexed)
        books_df (pd.DataFrame): 책 정보 DataFrame (컬럼 'book_id' 포함, 1-indexed)
        filename (str): 저장할 파일 이름
    """
    # 'user_item' 에지: shape=(2, num_edges)
    user_item_edges = edge_indices.get('user_item')
    if user_item_edges is None:
        raise ValueError("edge_indices에 'user_item' 키가 없습니다.")
    
    # Tensor에서 numpy 배열로 변환
    user_item_edges = user_item_edges.cpu().numpy()
    
    G = nx.Graph()
    
    # 사용자 노드 추가 (bipartite=0)
    num_users = users_df.shape[0]
    for i in range(num_users):
        # user_id는 1-indexed라고 가정
        user_id = users_df.iloc[i].get('user_id', i+1)
        G.add_node(f"user_{user_id}", bipartite=0)
    
    # 책 노드 추가 (bipartite=1)
    num_books = books_df.shape[0]
    for j in range(num_books):
        # book_id도 1-indexed라고 가정
        book_id = books_df.iloc[j].get('book_id', j+1)
        G.add_node(f"book_{book_id}", bipartite=1)
    
    # 사용자-책 에지 추가
    num_edges = user_item_edges.shape[1]
    for k in range(num_edges):
        # edge_indices에 저장된 값이 1-indexed라고 가정하므로, iloc에서 접근할 때는 1을 빼줍니다.
        u_idx = int(user_item_edges[0, k])
        b_idx = int(user_item_edges[1, k])
        
        user_id = users_df.iloc[u_idx - 1].get('user_id', u_idx)
        book_id = books_df.iloc[b_idx - 1].get('book_id', b_idx)
        
        G.add_edge(f"user_{user_id}", f"book_{book_id}")
    
    # 이분 그래프 레이아웃 계산 (bipartite_layout 사용)
    pos = nx.bipartite_layout(G, nodes=[f"user_{users_df.iloc[i].get('user_id', i+1)}" for i in range(num_users)])
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=["lightblue" if "user" in n else "lightgreen" for n in G.nodes()],
            node_size=500, font_size=8, edge_color="gray")
    plt.title("User-Book Graph")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_full_graph(node_features, edge_indices, users_df, books_df, user_vq, item_vq, filename="full_graph.png"):
    """
    전체 그래프 시각화: 사용자, 도서, 사용자 latent factor, 도서 latent factor를 모두 포함합니다.
    
    Args:
        node_features (dict): {'user': ..., 'item': ..., 'user_factor': ..., 'item_factor': ...}
        edge_indices (dict): {'user_item': ..., 'user_factor': ..., 'item_factor': ...}
        users_df (pd.DataFrame): 사용자 정보 (컬럼 'user_id' 존재, 1-indexed)
        books_df (pd.DataFrame): 도서 정보 (컬럼 'book_id' 존재, 1-indexed)
        user_vq (ResidualVectorQuantizer): 사용자 벡터 양자화 모델 (코드북 크기 정보 사용)
        item_vq (ResidualVectorQuantizer): 도서 벡터 양자화 모델
        filename (str): 저장할 파일 이름
    """
    G = nx.Graph()
    
    # 1. 사용자 노드 추가
    num_users = users_df.shape[0]
    for i in range(num_users):
        user_id = users_df.iloc[i].get("user_id", i+1)
        G.add_node(f"user_{user_id}", type="user")
    
    # 2. 도서 노드 추가
    num_books = books_df.shape[0]
    for j in range(num_books):
        book_id = books_df.iloc[j].get("book_id", j+1)
        G.add_node(f"book_{book_id}", type="book")
    
    # 3. 사용자 latent factor 노드 추가 (각 코드북에 대해)
    num_user_levels = len(user_vq.codebooks)
    codebook_size_user = user_vq.codebook_size
    for level in range(num_user_levels):
        for idx in range(codebook_size_user):
            G.add_node(f"user_factor_{level+1}_{idx}", type="user_factor")
    
    # 4. 도서 latent factor 노드 추가
    num_item_levels = len(item_vq.codebooks)
    codebook_size_item = item_vq.codebook_size
    for level in range(num_item_levels):
        for idx in range(codebook_size_item):
            G.add_node(f"book_factor_{level+1}_{idx}", type="book_factor")
    
    # 5. 사용자-도서 (user_item) 에지 추가
    if "user_item" in edge_indices:
        ui_edges = edge_indices["user_item"].cpu().numpy()
        num_edges = ui_edges.shape[1]
        for k in range(num_edges):
            # edge_indices가 1-indexed라고 가정
            u_idx = int(ui_edges[0, k])
            b_idx = int(ui_edges[1, k])
            G.add_edge(f"user_{u_idx}", f"book_{b_idx}", relation="user-item")
    
    # 6. 사용자-사용자 factor 에지 추가
    if "user_factor" in edge_indices:
        uf_edges = edge_indices["user_factor"].cpu().numpy()
        num_edges = uf_edges.shape[1]
        for k in range(num_edges):
            u_idx = int(uf_edges[0, k])    # 1-indexed 사용자
            f_idx = int(uf_edges[1, k])    # offset 포함 factor index
            level = f_idx // codebook_size_user + 1
            idx_in_level = f_idx % codebook_size_user
            G.add_edge(f"user_{u_idx}", f"user_factor_{level}_{idx_in_level}", relation="user-factor")
    
    # 7. 도서-도서 factor 에지 추가
    if "item_factor" in edge_indices:
        bf_edges = edge_indices["item_factor"].cpu().numpy()
        num_edges = bf_edges.shape[1]
        for k in range(num_edges):
            b_idx = int(bf_edges[0, k])    # 1-indexed 도서
            f_idx = int(bf_edges[1, k])    # offset 포함 factor index
            level = f_idx // codebook_size_item + 1
            idx_in_level = f_idx % codebook_size_item
            G.add_edge(f"book_{b_idx}", f"book_factor_{level}_{idx_in_level}", relation="book-factor")
    
    # 노드별 색상 지정
    color_map = {
        "user": "lightblue",
        "book": "lightgreen",
        "user_factor": "orange",
        "book_factor": "pink"
    }
    node_colors = [color_map.get(attr.get("type", ""), "gray") for n, attr in G.nodes(data=True)]
    
    # 레이아웃 계산 (spring layout)
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8)
    plt.title("Full User-Item Graph with Latent Factors")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()