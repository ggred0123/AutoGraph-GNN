# AutoGraph-GNN
# AutoGraph: LLM 기반 자동 그래프 구성 프레임워크

이 문서는 논문 "An Automatic Graph Construction Framework based on Large Language Models for Recommendation"을 기반으로 구현된 AutoGraph 프레임워크의 핵심 개념과 구현 방법에 대해 설명합니다.

## 1. 개요

AutoGraph는 대규모 언어 모델(LLM)을 활용하여 추천 시스템을 위한 그래프를 자동으로 구성하는 프레임워크입니다. 이 프레임워크는 두 가지 주요 한계점을 해결합니다:

1. **글로벌 뷰의 부재**: 기존 LLM 기반 그래프 구성 방법들은 전체 데이터셋의 특성을 고려하지 못합니다.
2. **구성 비효율성**: 기존 방법들은 O(N²) 시간 복잡도로 모든 노드 쌍을 비교해야 합니다.

## 2. 핵심 구성 요소

AutoGraph 프레임워크는 다음과 같은 핵심 구성 요소로 이루어져 있습니다:

### 2.1 의미 벡터 생성 (Semantic Vector Generation)

LLM을 활용하여 사용자와 아이템에 대한 풍부한 의미 벡터를 생성합니다.

- **사용자 의미 벡터**: 사용자 프로필과 상호작용 기록을 LLM에 입력하여 사용자의 선호도와 특성을 추론합니다.
- **아이템 의미 벡터**: 아이템 속성을 LLM에 입력하여 아이템의 콘텐츠와 특성을 분석합니다.

이 접근법의 핵심은 각 노드(사용자/아이템)에 대해 LLM을 한 번만 호출하는 **포인트와이즈 방식**을 사용하여 시간 복잡도를 O(N)으로 줄이는 것입니다.

### 2.2 잠재 요인 추출 (Latent Factor Extraction)

생성된 의미 벡터로부터 잠재 요인을 추출하기 위해 **잔차 양자화(Residual Quantization)**를 사용합니다.

- **코드북 구성**: 각 레벨마다 코드북(대표 벡터 집합)을 학습합니다.
- **양자화 과정**: 각 의미 벡터를 코드북 벡터들의 조합으로 표현합니다.
- **계층적 요인 추출**: 여러 레벨의 코드북을 사용하여 다양한 추상화 수준의 요인을 추출합니다.

이 과정을 통해 사용자와 아이템은 각각 잠재 요인 집합(K^u, K^i)으로 표현됩니다.

### 2.3 그래프 구성 (Graph Construction)

추출된 잠재 요인을 활용하여 고품질의 그래프를 구성합니다.

- **노드 유형**: 사용자(u), 아이템(i), 사용자 잠재 요인(q^u), 아이템 잠재 요인(q^i)
- **에지 유형**:
  - 사용자-아이템 에지(u-i): 사용자의 아이템 상호작용
  - 사용자-사용자 잠재 요인 에지(u-q^u): 사용자와 해당 잠재 요인 연결
  - 아이템-아이템 잠재 요인 에지(i-q^i): 아이템과 해당 잠재 요인 연결

이 그래프 구조는 의미적 유사성과 글로벌 특성 정보를 모두 포착할 수 있는 장점이 있습니다.

### 2.4 메타패스 기반 메시지 전파 (Metapath-based Message Propagation)

구성된 그래프에서 의미적 정보와 협업적 정보를 효과적으로 집계하기 위해 메타패스 기반 메시지 전파를 수행합니다.

- **의미적 메타패스**:
  - u→q→u (사용자→잠재요인→사용자): 의미적으로 유사한 사용자 간 연결
  - i→q→i (아이템→잠재요인→아이템): 의미적으로 유사한 아이템 간 연결

- **상호작용 메타패스**:
  - u→i (사용자→아이템): 사용자의 아이템 선호도
  - i→u (아이템→사용자): 아이템을 선호하는 사용자들

- **순차적 메시지 전파**:
  1. 의미적 메타패스를 통한 메시지 전파: H^u, H^i
  2. 상호작용 메타패스를 통한 메시지 전파: Ĥ^u, Ĥ^i

### 2.5 추천 향상 (Recommendation Enhancement)

그래프 강화 표현을 다양한 추천 모델에 통합하여 성능을 향상시킵니다.

- **모델 불가지론적(model-agnostic) 설계**: 어떤 추천 모델과도 쉽게 통합될 수 있음
- **보조 특성 활용**: 그래프 강화 표현(Ĥ^u, Ĥ^i)을 추가 특성으로 사용
- **선호도 점수 계산**: Φ(S^u, F^u, F^i, Ĥ^u, Ĥ^i)

## 3. 구현 상세

### 3.1 SemanticVectorGenerator 클래스

LLM을 활용하여 사용자와 아이템의 의미 벡터를 생성합니다.

```python
# 사용자 프롬프트 예시
prompt = f"Given a {gender} user who is aged {age}, this user's viewing "\
         f"history over time is listed below:\n{history_str}. "\
         f"Analyze the user's preferences..."
```

### 3.2 ResidualVectorQuantizer 클래스

잔차 양자화를 통해 의미 벡터에서 잠재 요인을 추출합니다.

```python
# 양자화 과정
for i in range(self.num_codebooks):
    # 현재 코드북
    codebook = self.codebooks[i]
    
    # 가장 가까운 코드북 벡터 찾기
    distances = torch.cdist(residual, codebook)
    min_indices = torch.argmin(distances, dim=1)
    
    # 선택된 코드북 벡터
    selected_vectors = codebook[min_indices]
    
    # 잔차 업데이트
    residual = residual - selected_vectors
```

### 3.3 AutoGraphConstructor 클래스

추출된 잠재 요인을 활용하여 그래프를 구성합니다.

```python
# 사용자-사용자 잠재 요인 에지 구성
user_factor_edges = []
for user_id, user_factor_ids in enumerate(zip(*user_indices)):
    for level, factor_id in enumerate(user_factor_ids):
        user_factor_edges.append([user_id, factor_id.item() + level * self.user_vq.codebook_size])
```

### 3.4 MetapathGNN 클래스

메타패스 기반 메시지 전파를 수행하여 그래프 강화 표현을 생성합니다.

```python
# 1단계: 의미적 메타패스 집계
u_q_u_emb = self.gat_u_q_u(all_embs, metapath_edge_indices['u_q_u'])
i_q_i_emb = self.gat_i_q_i(all_embs, metapath_edge_indices['i_q_i'])

# 2단계: 상호작용 메타패스 집계
H_hat_u = self.gat_i_u(combined_emb, metapath_edge_indices['i_u'])[:num_users]
H_hat_i = self.gat_u_i(combined_emb, metapath_edge_indices['u_i'])[num_users:num_users+num_items]
```

### 3.5 AutoGraphRecommender 클래스

그래프 강화 표현을 활용하여 최종 선호도 점수를 계산합니다.

```python
# 원본 특성과 그래프 강화 임베딩 결합
combined = torch.cat([user_enc, item_enc, user_graph_emb, item_graph_emb], dim=1)

# 최종 점수 계산
score = self.combiner(combined)
```

## 4. 성능 및 효율성

### 4.1 시간 복잡도 개선

- **기존 방법**: O(N²) - 모든 노드 쌍에 대해 LLM 호출
- **AutoGraph**: O(N) - 각 노드마다 한 번씩 LLM 호출

### 4.2 성능 향상

논문에서 보고된 실험 결과에 따르면:

- 다양한 데이터셋과 백본 모델에 대해 최첨단 성능 달성
- 산업용 광고 플랫폼에서 RPM 2.69% 개선, eCPM 7.31% 개선
- 온라인 A/B 테스트에서 유의미한 개선 확인

## 5. 결론

AutoGraph는 LLM의 의미적 지식을 활용하면서도 효율적이고 글로벌 관점을 유지하는 그래프 구성 프레임워크입니다. 잠재 요인 추출을 통해 시간 복잡도를 개선하고, 메타패스 기반 메시지 전파를 통해 의미적 정보와 협업적 정보를 효과적으로 통합합니다. 이 프레임워크는 다양한 추천 모델과 쉽게 통합될 수 있으며, 실제 산업 환경에서도 유의미한 성능 향상을 제공합니다.
