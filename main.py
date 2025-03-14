import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False    
from tqdm import tqdm

# 앞서 작성한 모듈 임포트

from models.semantic_generator import BookSemanticVectorGenerator
from models.vector_quantizer import ResidualVectorQuantizer
from pipeline import AutoGraphPipeline  
from configs.default_config import get_config
from models.graph_constructor import AutoGraphConstructor
from models.metapath_gnn import MetaPathGNN
from models.recommender import AutoGraphRecommender
from utils.visualization import visualize_residual_quantization, visualize_full_graph

def main():
    """
    AutoGraph를 이용한 도서 추천 시스템 구현 메인 함수
    """
    print("=== AutoGraph 기반 도서 추천 시스템 ===")
    
 
    
    # 이미 데이터가 있으면 로드, 없으면 생성
    data_dir = "./data"
    if os.path.exists(f"{data_dir}/books.csv") and os.path.exists(f"{data_dir}/users.csv"):
        print("기존 데이터셋을 로드합니다...")
        books_df = pd.read_csv(f"{data_dir}/books.csv")
        users_df = pd.read_csv(f"{data_dir}/users.csv")
        interactions_df = pd.read_csv(f"{data_dir}/interactions.csv")
    
    # 데이터셋 정보 출력
    print("\n=== 데이터셋 정보 ===")
    print(f"도서 수: {len(books_df)}")
    print(f"사용자 수: {len(users_df)}")
    print(f"상호작용 수: {len(interactions_df)}")
    
    # 2. 트랜스포머 모델로 의미 벡터 생성
    print("\n2. 트랜스포머 모델로 책 의미 벡터 생성 중...")
    semantic_generator = BookSemanticVectorGenerator("sentence-transformers/all-MiniLM-L6-v2")
    
    # 책 임베딩 생성
    book_vectors = semantic_generator.generate_book_embeddings(
        books_df, batch_size=8, combine_method='concat'
    )
    print(f"책 의미 벡터 형태: {book_vectors.shape}")
    
    # 사용자 임베딩 생성
    user_vectors = semantic_generator.generate_user_embeddings(
        users_df, interactions_df, books_df, method='weighted', batch_size=8
    )
    print(f"사용자 의미 벡터 형태: {user_vectors.shape}")
    
    # 3. 벡터 양자화를 통한 잠재 요인 추출
    print("\n3. 벡터 양자화로 잠재 요인 추출 중...")
    
    # 책 벡터 양자화 모델 설정
    book_vq = ResidualVectorQuantizer(
        input_dim=book_vectors.shape[1],  # 의미 벡터 차원
        hidden_dim=64,                   # 내부 표현 차원
        codebook_size=32,                # 각 코드북 크기
        num_codebooks=3                  # 코드북 레벨 수
    )
    
    # 사용자 벡터 양자화 모델 설정
    user_vq = ResidualVectorQuantizer(
        input_dim=user_vectors.shape[1],  # 의미 벡터 차원
        hidden_dim=64,                   # 내부 표현 차원
        codebook_size=16,                # 각 코드북 크기 (사용자는 더 적은 수로 설정)
        num_codebooks=2                  # 코드북 레벨 수
    )
    
    # 학습 루프
    print("벡터 양자화 모델 학습 중...")
    num_epochs = 100
    learning_rate = 1e-3
    
    book_optimizer = torch.optim.Adam(book_vq.parameters(), lr=learning_rate)
    user_optimizer = torch.optim.Adam(user_vq.parameters(), lr=learning_rate)
    
    # 책 벡터 양자화 모델 학습
    book_losses = []
    for epoch in range(num_epochs):
        # 순방향 전파
        book_quantized, book_indices, book_reconstructed, book_loss = book_vq(book_vectors)
        
        # 역전파
        book_optimizer.zero_grad()
        book_loss.backward()
        book_optimizer.step()
        
        book_losses.append(book_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, 책 손실: {book_loss.item():.6f}")
    
    # 사용자 벡터 양자화 모델 학습
    user_losses = []
    for epoch in range(num_epochs):
        # 순방향 전파
        user_quantized, user_indices, user_reconstructed, user_loss = user_vq(user_vectors)
        
        # 역전파
        user_optimizer.zero_grad()
        user_loss.backward()
        user_optimizer.step()
        
        user_losses.append(user_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, 사용자 손실: {user_loss.item():.6f}")
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(book_losses)
    plt.title('책 벡터 양자화 학습 곡선')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(user_losses)
    plt.title('사용자 벡터 양자화 학습 곡선')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png")
    
    # 4. 코드북 시각화 및 분석
    print("\n4. 코드북 분석 중...")
    book_vq.visualize_codebooks("book_codebooks.png")
    user_vq.visualize_codebooks("user_codebooks.png")

    visualize_residual_quantization(input_dim=book_vectors.shape[1],
                                hidden_dim=64,
                                codebook_size=32,
                                num_codebooks=3,
                                num_samples=100)
    
    # 코드북 사용 통계
    book_usage_stats = book_vq.get_codebook_usage()
    user_usage_stats = user_vq.get_codebook_usage()
    
    print("\n=== 책 코드북 사용 통계 ===")
    for stats in book_usage_stats:
        print(f"레벨 {stats['level']}: 활성화 비율 = {stats['activation_ratio']:.2f}, " 
              f"유효 크기 = {stats['effective_size']}/{book_vq.codebook_size}")
    
    print("\n=== 사용자 코드북 사용 통계 ===")
    for stats in user_usage_stats:
        print(f"레벨 {stats['level']}: 활성화 비율 = {stats['activation_ratio']:.2f}, " 
              f"유효 크기 = {stats['effective_size']}/{user_vq.codebook_size}")
    
    # 5. 양자화 인덱스 추출
    print("\n5. 양자화 인덱스 추출 중...")
    with torch.no_grad():
        book_quantized, book_indices, _, _ = book_vq(book_vectors)
        user_quantized, user_indices, _, _ = user_vq(user_vectors)
    
    # 양자화 인덱스를 DataFrame으로 변환하여 분석
    book_factors_df = pd.DataFrame()
    for level, indices in enumerate(book_indices):
        book_factors_df[f'level_{level+1}_factor'] = indices.numpy()
    
    book_factors_df['book_id'] = books_df['book_id'].values
    book_factors_df['title'] = books_df['title'].values
    book_factors_df['genre'] = books_df['genre'].values
    
    user_factors_df = pd.DataFrame()
    for level, indices in enumerate(user_indices):
        user_factors_df[f'level_{level+1}_factor'] = indices.numpy()
    
    user_factors_df['user_id'] = users_df['user_id'].values
    
    # 결과 저장
    book_factors_df.to_csv(f"{data_dir}/book_factors.csv", index=False)
    user_factors_df.to_csv(f"{data_dir}/user_factors.csv", index=False)
    
    # 6. 장르별 잠재 요인 분포 분석
    print("\n6. 장르별 잠재 요인 분포 분석 중...")
    
    # 레벨 1 잠재 요인 분석 (가장 일반적인 특성)
    level1_factor_col = 'level_1_factor'
    genre_factor_counts = pd.crosstab(book_factors_df['genre'], book_factors_df[level1_factor_col])
    
    # 열 합계로 정규화
    genre_factor_dist = genre_factor_counts.div(genre_factor_counts.sum(axis=0), axis=1)
    
    # 히트맵 시각화
    plt.figure(figsize=(12, 8))
    plt.title('장르별 레벨 1 잠재 요인 분포')
    sns.heatmap(genre_factor_dist, cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': '비율'})
    plt.tight_layout()
    plt.savefig("genre_factor_distribution.png")
    
    print("\n모든 과정이 완료되었습니다!")
    print("생성된 파일:")
    print("- training_curves.png: 벡터 양자화 학습 곡선")
    print("- book_codebooks.png: 책 코드북 시각화")
    print("- user_codebooks.png: 사용자 코드북 시각화")
    print("- genre_factor_distribution.png: 장르별 잠재 요인 분포")
    print(f"- {data_dir}/book_factors.csv: 책 잠재 요인")
    print(f"- {data_dir}/user_factors.csv: 사용자 잠재 요인")
    
    
    
    # 7. 그래프 구성
    print("\n7. 그래프 구성 중...")
    # 그래프 구성기 임포트 (graph_constructor.py 구현 필요)
    from models.graph_constructor import AutoGraphConstructor

    # 그래프 구성기 초기화
    graph_constructor = AutoGraphConstructor(book_vq, user_vq)

    # 그래프 구성
    graph_data = graph_constructor.construct_graph(
        book_vectors, user_vectors, 
        list(zip(interactions_df['user_id'], interactions_df['book_id']))
    )

    # 8. 메타패스 기반 메시지 전파
    print("\n8. 메타패스 기반 메시지 전파 중...")
    # 메타패스 기반 GNN 모듈 임포트 (metapath_gnn.py 구현 필요)

    # 메타패스 GNN 초기화
    metapath_gnn = MetaPathGNN(
        user_dim=user_vectors.shape[1],
        item_dim=book_vectors.shape[1],
        factor_dim=64,  # hidden_dim과 동일
        hidden_dim=64,
        num_heads=4
    )
    node_features, edge_indices = graph_constructor.construct_graph(
    book_vectors, user_vectors, 
    list(zip(interactions_df['user_id'], interactions_df['book_id']))
)
    

    # 메타패스 에지 준비
    metapath_edge_indices = graph_constructor.prepare_metapath_edges(edge_indices)

    # 사용자-책 그래프 시각화 (NetworkX 사용)
    visualize_full_graph(node_features, edge_indices, users_df, books_df, user_vq, book_vq, filename="full_user_item_graph.png")


    # 메시지 전파 수행
    book_graph_emb, user_graph_emb = metapath_gnn(
        node_features, 
        metapath_edge_indices
    )

    # 9. 추천 생성
    print("\n9. 추천 생결과 생성 중...")
    # 추천 모델 임포트 (recommender.py 구현 필요)
    from models.recommender import AutoGraphRecommender

    # 추천 모델 초기화
    recommender = AutoGraphRecommender(
        user_dim=user_vectors.shape[1],
        item_dim=book_vectors.shape[1],
        hidden_dim=64,
        output_dim=1
    )

    # 테스트 사용자에 대한 추천 생성
    test_user_id = 1  # 첫 번째 사용자
    test_user_idx = test_user_id - 1
    test_user_vector = user_vectors[test_user_idx].unsqueeze(0)
    test_user_graph_emb = user_graph_emb[test_user_idx].unsqueeze(0)

    # 모든 책에 대한 점수 계산 (여기서 book_graph_emb 사용)
    scores = []
    for i in range(len(books_df)):
        book_vector = book_vectors[i].unsqueeze(0)
        book_graph_emb_i = book_graph_emb[i].unsqueeze(0)
        
        score = recommender(
            test_user_vector, 
            book_vector, 
            test_user_graph_emb, 
            book_graph_emb_i
        )
        scores.append(score.item())

    # 점수 기반 추천 리스트 생성
    recommendations = pd.DataFrame({
        'book_id': books_df['book_id'],
        'title': books_df['title'],
        'author': books_df['author'],
        'genre': books_df['genre'],
        'score': scores
    })

    # 이미 읽은 책 필터링
    user_read_books = interactions_df[interactions_df['user_id'] == test_user_id]['book_id'].values
    recommendations['already_read'] = recommendations['book_id'].isin(user_read_books)

    # 점수 내림차순으로 정렬하고 읽지 않은 책만 추천
    top_recommendations = recommendations[~recommendations['already_read']].sort_values('score', ascending=False).head(10)

    print(f"\n사용자 {test_user_id}에 대한 상위 추천 도서:")
    for i, (_, book) in enumerate(top_recommendations.iterrows(), 1):
        print(f"{i}. {book['title']} (장르: {book['genre']}, 점수: {book['score']:.4f})")

    # 추천 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(top_recommendations['title'], top_recommendations['score'])
    plt.title(f'사용자 {test_user_id}에 대한 상위 추천 도서')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('추천 점수')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("recommendations.png")

    print("\n추천 결과가 recommendations.png 파일로 저장되었습니다.")
    
    # 반환값 (필요한 경우 사용)
    return {
        'books_df': books_df,
        'users_df': users_df,
        'interactions_df': interactions_df,
        'book_vectors': book_vectors,
        'user_vectors': user_vectors,
        'book_vq': book_vq,
        'user_vq': user_vq,
        'book_factors_df': book_factors_df,
        'user_factors_df': user_factors_df
    }


if __name__ == "__main__":
    # matplotlib 설정
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 이미지 파일로 저장
    import seaborn as sns
    
    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 메인 함수 실행
    main()