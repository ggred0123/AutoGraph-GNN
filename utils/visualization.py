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
