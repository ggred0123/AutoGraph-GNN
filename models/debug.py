import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from vector_quantizer import ResidualVectorQuantizer



from sklearn.cluster import KMeans

def kmeans_initialize_vq(model, data, device='cuda'):
    """
    model: ResidualVectorQuantizer 인스턴스
    data: 학습에 사용할 입력 데이터 (torch.Tensor, shape: (N, input_dim))
    
    각 레벨별로 encoder의 출력에서 잔차를 계산하여 k-means로 클러스터링하고,
    그 클러스터 중심을 해당 레벨의 codebook 초기값으로 설정합니다.
    """
    model.eval()  # 초기화 시엔 eval 모드로 진행 (모델 파라미터 업데이트 X)
    with torch.no_grad():
        # encoder를 통해 초기 은닉 벡터 h 얻기
        h = model.encoder(data.to(device))
        residual = h.clone()
        
        for i in range(model.num_codebooks):
            # 현재 레벨의 잔차(residual)를 numpy 배열로 변환
            X = residual.cpu().numpy()
            kmeans = KMeans(n_clusters=model.codebook_size, n_init=10, random_state=42)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_  # shape: (codebook_size, hidden_dim)
            
            # 해당 레벨의 codebook을 클러스터 중심으로 초기화
            with torch.no_grad():
                model.codebooks[i].copy_(torch.tensor(centroids, 
                                                       dtype=model.codebooks[i].dtype, 
                                                       device=model.codebooks[i].device))
            
            # 현재 레벨에 대해 k-means 클러스터 할당 결과(labels)를 통해 잔차 계산
            assignments = kmeans.labels_  # shape: (N,)
            centroids_tensor = torch.tensor(centroids, device=residual.device, dtype=residual.dtype)
            # 각 샘플에 대해 할당된 클러스터의 중심 벡터 선택
            selected = centroids_tensor[assignments]
            # 잔차 업데이트: 현재 residual에서 선택된 벡터를 빼줌
            residual = residual - selected
    model.train()  # 초기화 후 다시 train 모드로 전환


def debug_small_vq_test():
    # 1) 작은 synthetic data 100개
    input_dim = 64
    x = torch.randn(1000, input_dim)  # (100, 64)
    
    # 2) VQ 모델 생성
    model = ResidualVectorQuantizer(
        input_dim=input_dim,
        hidden_dim=32,
        codebook_size=8,
        num_codebooks=2
    )
    
    kmeans_initialize_vq(model, x, device='cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3) DataLoader 구성
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 4) 훈련 루프 + 디버그
    for epoch in range(50):
        for batch_idx, (batch_x,) in enumerate(loader):
            # forward
            quantized, indices, recon, loss = model(batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
    
    # 5) get_codebook_usage 등 찍기
    usage = model.get_codebook_usage()
    for s in usage:
        print(s)

if __name__=="__main__":
    debug_small_vq_test()
