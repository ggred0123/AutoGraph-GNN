import torch 
import torch.nn as nn
import torch.nn.functional as F

class ResidualVectorQuantizer(nn.Module):
    """
    잔차 양자화를 이용한 벡터 양자화 모듈입니다.
    """
    def __init__(self, input_dim, hidden_dim, codebook_size, num_codebooks=3):
        super(ResidualVectorQuantizer, self).__init__() # 부모 클래스의 생성자 호출
        
        self.input_dim = input_dim # 입력 차원
        self.hidden_dim = hidden_dim # 인코더 출력 차원
        self.codebook_size = codebook_size # 코드북 크기
        self.num_codebooks = num_codebooks # 코드북 수
        
        self.encoder == nn.Sequential( # 인코더 정의
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
          
        )
        self.decoder = nn.Sequential( # 디코더 정의
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, input_dim),
        )
        
        self.codebooks = nn.ParameterList([ # 코드북 정의
            nn.Parameter(torch.randn(codebook_size, hidden_dim)) # 코드북 초기화
            for _ in range(num_codebooks) # 코드북 수만큼 반복
        ])
    
    def forward(self, x):
        """
        순방향 전파 함수
        Args:
            x (torch.Tensor): 입력 벡터
            
        Returns:
            torch.Tensor: 양자화된 벡터
        """
        h = self.encoder(x) #인코더를 통해 히든 벡터 얻기
        
        residual = h.clone()
        
        indices = [] #각 코드북에서 가장 가까운 벡터의 인덱스를 저장할 리스트
        codebook_vectors = [] #각 코드북에서 가장 가까운 벡터를 저장할 리스트
        
        for i in range(self.num_codebooks): # 순차적 양자화 수행
            
            codebook = self.codebooks[i] # 현재 코드북 선택
            
            distances = torch.cdist(residual, codebook) #코드북과 히든 벡터 간의 유클리드 거리 계산
            
            min_indices = torch.argmin(distances, dim =1) # 가장 가까운 벡터의 인덱스 찾기
            
            indices.append(min_indices) # 현재 코드북에서 가장 가까운 벡터의 인덱스 저장
            
            selected_vectors = codebook[min_indices] # 선택된 벡터 추출
            codebook_vectors.append(selected_vectors) # 선택된 벡터 저장
            
            residual = residual - selected_vectors # 잔차 계산 update
            
        quantized = sum(codebook_vectors) # 모든 코드북에서 선택된 벡터들의 합
        reconstructed = self.decoder(quantized) # 양자화된 벡터를 디코더에 통과시켜 재구성 벡터 얻기
        
        
        if self.training:
            
            rec_loss = F.mse_loss(reconstructed, x) # 재구성 오차 계산
            
            com_loss =0 #commitment loss를 각 레벨에 대해 계산
            
            for i in range(self.num_codebooks):
                residual_i = h.clone() - sum(codebook_vectors[:i]) if i > 0 else h.clone()
                selected_vectors_i = codebook_vectors[i] # 현재 코드북에서 선택된 벡터
                
                com_loss += F.mse_loss(residual_i.detach(), selected_vectors_i) # 잔차가 코드북 벡터에 가까워지도록
                
                com_loss += F.mse_loss(residual_i, selected_vectors_i.detach()) # 코드북 벡터가 잔차에 가까워지도록

            loss = rec_loss + com_loss * 0.25 #commitment loss에 가중치 0.25 적용
        else:
            loss  = None
        return quantized, indices, reconstructed, loss
    
    @torch.no_grad() # 파라미터 업데이트 없이 인코딩 수행
    def encode(self, x):
        """
        인코딩 함수
        입력 벡터르 코드북 인덱스로 인코딩
        """
        h = self.encoder(x) # 인코더를 통해 히든 벡터 얻기
        
        residual = h.clone() # 잔차 초기화
        
        indices = [] # 각 코드북에서 가장 가까운 벡터의 인덱스를 저장할 리스트
        
        for i in range(self.num_codebooks): # 순차적 양자화 수행
            codebook = self.codebooks[i] # 현재 코드북 선택
            
            distances = torch.cdist(residual, codebook) # 코드북과 히든 벡터 간의 유클리드 거리 계산
            
            min_indices = torch.argmin(distances, dim =1) # 가장 가까운 벡터의 인덱스 찾기
            
            indices.append(min_indices) # 현재 코드북에서 가장 가까운 벡터의 인덱스 저장
            
            selected_vectors = codebook[min_indices] # 선택된 벡터 추출
            residual = residual - selected_vectors # 잔차 계산 update
            
        return indices
    

        
            
            
        
       
        
            
        