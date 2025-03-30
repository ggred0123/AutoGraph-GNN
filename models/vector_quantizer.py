import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np



class ResidualVectorQuantizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size, num_codebooks=3):
        super(ResidualVectorQuantizer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )
        
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, hidden_dim))
            for _ in range(num_codebooks)
        ])
        
        # 마지막 forward에서 계산한 인덱스를 저장할 속성을 초기화합니다.
        self.last_indices = None

    def forward(self, x):
        h = self.encoder(x)
        residual = h.clone()
        indices = []
        codebook_vectors = []
        for i in range(self.num_codebooks):
            codebook = self.codebooks[i]
            distances = torch.cdist(residual, codebook)
            min_indices = torch.argmin(distances, dim=1)
            indices.append(min_indices)
            selected_vectors = codebook[min_indices]
            codebook_vectors.append(selected_vectors)
            residual = residual - selected_vectors
        # 마지막 계산된 인덱스를 저장합니다.
        self.last_indices = indices
        
        quantized = sum(codebook_vectors)
        reconstructed = self.decoder(quantized)
        
        if self.training:
            rec_loss = F.mse_loss(reconstructed, x)
            com_loss = 0
            for i in range(self.num_codebooks):
                residual_i = h.clone() - sum(codebook_vectors[:i]) if i > 0 else h.clone()
                selected_vectors_i = codebook_vectors[i]
                com_loss += F.mse_loss(residual_i.detach(), selected_vectors_i)
                com_loss += F.mse_loss(residual_i, selected_vectors_i.detach())
            loss = rec_loss + com_loss * 0.5
        else:
            loss = None
        return quantized, indices, reconstructed, loss

    def get_codebook_usage(self):
        """
        마지막 forward pass에서 각 코드북 레벨의 사용 현황을 계산합니다.
        
        Returns:
            List[dict]: 각 레벨에 대한 사용 통계 (레벨, activation_ratio, effective_size)
        """
        if self.last_indices is None:
            raise ValueError("No forward pass data available. Run forward() first.")
        usage_stats = []
        for i, indices in enumerate(self.last_indices):
            unique_codes = torch.unique(indices)
            effective_size = unique_codes.numel()
            activation_ratio = effective_size / self.codebook_size
            usage_stats.append({
                'level': i + 1,
                'activation_ratio': activation_ratio,
                'effective_size': effective_size
            })
        return usage_stats

    def visualize_codebooks(self, filename):
        """
        각 코드북을 2차원 공간으로 투영하여 시각화한 후, filename으로 저장합니다.
        """
        num_codebooks = len(self.codebooks)
        hidden_dim = self.codebooks[0].shape[1]
        pca = PCA(n_components=2)
        plt.figure(figsize=(6 * num_codebooks, 6))
        for i, codebook in enumerate(self.codebooks):
            codebook_np = codebook.detach().cpu().numpy()  # (codebook_size, hidden_dim)
            if hidden_dim > 2:
                codebook_2d = pca.fit_transform(codebook_np)
            else:
                codebook_2d = codebook_np
            plt.subplot(1, num_codebooks, i + 1)
            plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], s=40, alpha=0.7)
            plt.title(f"Codebook Level {i+1}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def kmeans_initialize_vq(model, data, device=None):
        """
        모델의 각 레벨에 대해 encoder 출력의 residual을 kmeans로 클러스터링하여
        codebook을 초기화합니다.
        
        만약 반환된 클러스터 수(actual_clusters)가 model.codebook_size(desired_clusters)보다 작으면,
        부족한 centroids는 반복하여 패딩(padding)합니다.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()  # 초기화 시 eval 모드
        with torch.no_grad():
            h = model.encoder(data.to(device))
            residual = h.clone()
            
            for i in range(model.num_codebooks):
                X = residual.cpu().numpy()
                n_samples = X.shape[0]
                desired_clusters = model.codebook_size
                # n_clusters는 단순히 min(desired_clusters, n_samples)로 설정
                n_clusters = desired_clusters if n_samples >= desired_clusters else n_samples
                
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                kmeans.fit(X)
                centroids = kmeans.cluster_centers_  # shape: (actual_clusters, hidden_dim)
                
                # 실제 클러스터 수 확인 후, 부족한 경우 패딩
                actual_clusters = centroids.shape[0]
                if actual_clusters < desired_clusters:
                    pad_size = desired_clusters - actual_clusters
                    # 예시로 centroids의 앞부분을 반복하여 패딩합니다.
                    pad_centroids = centroids[:pad_size]
                    centroids = np.vstack([centroids, pad_centroids])
                
                assert centroids.shape[0] == desired_clusters, f"Expected {desired_clusters} centroids, got {centroids.shape[0]}"
                
                # 해당 레벨의 codebook을 kmeans 결과로 초기화
                model.codebooks[i].copy_(torch.tensor(centroids, 
                                                        dtype=model.codebooks[i].dtype, 
                                                        device=model.codebooks[i].device))
                
                # 잔차 업데이트: 각 샘플에 대해 kmeans 할당 결과에 따라
                assignments = kmeans.labels_
                centroids_tensor = torch.tensor(centroids, device=residual.device, dtype=residual.dtype)
                assignments_tensor = torch.tensor(assignments, device=residual.device)
                selected = centroids_tensor[assignments_tensor]
                residual = residual - selected
        model.train()  # 초기화 후 train 모드로 전환