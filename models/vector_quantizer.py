import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
            loss = rec_loss + com_loss * 0.25
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
