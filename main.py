import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import torch
import numpy as np
import pandas as pd
import json
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
from models.like_weight_processor import LikesWeightProcessor

def main():
    """
    AutoGraph를 이용한 도서 추천 시스템 구현 메인 함수
    """
    print("=== AutoGraph 기반 도서 추천 시스템 ===")
    
 
    
    # 임베딩 파일만 사용
    data_dir = "./data"
    print("저장된 책 임베딩 파일만 사용합니다...")
            
            
            
    print("임베딩 파일을 로드하는 중...")
    # embedding_ids.json 로드
    with open(f"{data_dir}/embedding_ids.json", 'r') as f:
        book_ids = json.load(f)
    
    # embeddings.npy 로드
    book_vectors = np.load(f"{data_dir}/embeddings.npy")
    
    print(f"책 임베딩 수: {len(book_ids)}")
    print(f"책 의미 벡터 형태: {book_vectors.shape}")
    
    # 데이터프레임 생성 (ID만 있는 간단한 형태)
    books_df = pd.DataFrame({'book_id': book_ids})
    
    # API에서 사용자 ID 및 좋아요 데이터 가져오기
    print("API에서 사용자 데이터 가져오는 중...")
    import requests
    
    try:
        response = requests.get("https://flik-919620445413.asia-northeast1.run.app/users")
        response.raise_for_status()  # 에러 체크
        users_data = response.json()
        
        # 사용자 ID 추출
        user_ids = [user["id"] for user in users_data]
        
        # 중복 제거 및 정렬
        user_ids = sorted(list(set(user_ids)))
        
        # 사용자 데이터프레임 생성
        users_df = pd.DataFrame({'user_id': user_ids})
        print(f"API에서 가져온 사용자 수: {len(users_df)}")
        
        # 좋아요 데이터 추출
        likes = []
        for user in users_data:
            user_id = user["id"]
            for book_id in user.get("likedBookIds", []):
                # book_id가 우리가 가지고 있는 책 ID에 있는 경우만 추가
                if book_id in book_ids:
                    likes.append({'user_id': user_id, 'book_id': book_id})
        
        likes_df = pd.DataFrame(likes) if likes else pd.DataFrame(columns=['user_id', 'book_id'])
        print(f"API에서 가져온 좋아요 수: {len(likes_df)}")
        
        # 좋아요 데이터가 없으면 가상 데이터 생성
        if len(likes_df) == 0:
            print("경고: API에서 가져온 좋아요 데이터가 없습니다. 가상 데이터를 생성합니다.")
            # 가상 좋아요 데이터 생성
            for user_id in users_df['user_id'].tolist():
                # 각 사용자는 0~3개의 책을 좋아함
                n_likes = np.random.randint(0, 4)
                if n_likes > 0:  # 좋아요가 있는 경우만 처리
                    # 랜덤 책 선택
                    liked_books = np.random.choice(book_ids, size=min(n_likes, len(book_ids)), replace=False)
                    for book_id in liked_books:
                        likes.append({'user_id': user_id, 'book_id': book_id})
            
            likes_df = pd.DataFrame(likes)
            print(f"가상 좋아요 수: {len(likes_df)}")
        
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        print("가상 사용자와 좋아요 데이터를 사용합니다.")
        # 가상 사용자 생성
        num_users = 10
        users_df = pd.DataFrame({'user_id': range(1, num_users + 1)})
        
        # 가상 좋아요 데이터 생성
        likes = []
        for user_id in users_df['user_id']:
            # 각 사용자는 0~3개의 책을 좋아함
            n_likes = np.random.randint(0, 4)
            if n_likes > 0:  # 좋아요가 있는 경우만 처리
                # 랜덤 책 선택
                liked_books = np.random.choice(book_ids, size=min(n_likes, len(book_ids)), replace=False)
                for book_id in liked_books:
                    likes.append({'user_id': user_id, 'book_id': book_id})
        
        likes_df = pd.DataFrame(likes)
        print(f"가상 좋아요 수: {len(likes_df)}")
    
    # 사용자 임베딩을 생성하지 않고 랜덤한 벡터로 대체
    user_dim = book_vectors.shape[1]  # 책 벡터와 같은 디멘전 사용
    num_users = len(users_df)
    user_vectors = np.random.randn(num_users, user_dim).astype(np.float32)  # 가상의 사용자 임베딩
    print(f"사용자 임베딩 생성: {user_vectors.shape}")
    
    # 상호작용 데이터 생성 (좋아요 데이터를 일부 포함하여 생성)
    interactions = []
    
    # 좋아요 데이터를 전부 상호작용으로 추가 (좋아요한 책은 반드시 읽었다고 가정)
    # 좋아요 데이터가 있는 경우
    if not likes_df.empty:
        interactions.extend(likes_df.to_dict('records'))
    
    # 추가로 랜덤 상호작용 생성
    for user_id in users_df['user_id'].tolist():
        # 추가로 1~3개의 책과 상호작용
        n_extra = np.random.randint(1, 4)
        # 이미 좋아요한 책은 제외한 가능한 동귀 목록
        liked_books = likes_df[likes_df['user_id'] == user_id]['book_id'].tolist() if not likes_df.empty else []
        available_books = list(set(book_ids) - set(liked_books))
        
        if available_books and n_extra > 0:
            # 가능한 책 수가 n_extra보다 적은 경우 처리
            n_select = min(n_extra, len(available_books))
            if n_select > 0:
                books = np.random.choice(available_books, size=n_select, replace=False)
                for book_id in books:
                    interactions.append({'user_id': user_id, 'book_id': book_id})
    
    interactions_df = pd.DataFrame(interactions)
    print(f"상호작용 데이터 수: {len(interactions_df)}")
    

    
    # 임베딩과 가상 데이터를 기반으로 책과 사용자 임베딩 처리
    print("\n임베딩 처리 중...")
    like_weight_processor = LikesWeightProcessor(alpha=0.7)
    book_vectors, user_vectors = like_weight_processor.apply_likes_weights(book_vectors, user_vectors, likes_df)
    
    
    # 3. 벡터 양자화를 통한 잠재 요인 추출
    print("\n3. 벡터 양자화로 잠재 요인 추출 중...")
    
    # NumPy 배열이 확실하게 전달되도록 book_vectors와 user_vectors를 복사
    book_vectors_array = book_vectors
    user_vectors_array = user_vectors
    

    # 책 벡터 양자화 모델 설정
    book_vq = ResidualVectorQuantizer(
        input_dim=book_vectors.shape[1],  # 의미 벡터 차원
        hidden_dim=64,                   # 내부 표현 차원
        codebook_size=8,                # 각 코드북 크기
        num_codebooks=3                  # 코드북 레벨 수
    )
    ResidualVectorQuantizer.kmeans_initialize_vq(book_vq, book_vectors_array, device='cpu')
    
    print()
    # 사용자 벡터 양자화 모델 설정
    user_vq = ResidualVectorQuantizer(
        input_dim=user_vectors.shape[1],  # 의미 벡터 차원
        hidden_dim=64,                   # 내부 표현 차원
        codebook_size=8,                # 각 코드북 크기 (사용자는 더 적은 수로 설정)
        num_codebooks=2                  # 코드북 레벨 수
    )
    ResidualVectorQuantizer.kmeans_initialize_vq(user_vq, user_vectors_array, device='cpu')
    
    
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
        book_quantized, book_indices, book_reconstructed, book_loss = book_vq(book_vectors_array)
        
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
        user_quantized, user_indices, user_reconstructed, user_loss = user_vq(user_vectors_array)
        
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
        book_quantized, book_indices, _, _ = book_vq(book_vectors_array)
        user_quantized, user_indices, _, _ = user_vq(user_vectors_array)
    
    # 양자화 인덱스를 DataFrame으로 변환하여 분석
    book_factors_df = pd.DataFrame()
    for level, indices in enumerate(book_indices):
        book_factors_df[f'level_{level+1}_factor'] = indices.numpy()
    
    # 책 ID 추가
    book_factors_df['book_id'] = books_df['book_id'].values
    
    # title과 genre 컬럼이 있는지 확인하고 없으면 빈 값 추가
    if 'title' in books_df.columns:
        book_factors_df['title'] = books_df['title'].values
    else:
        # book_id를 임시 제목으로 사용
        book_factors_df['title'] = [f'Book {id}' for id in books_df['book_id'].values]
    
    if 'genre' in books_df.columns:
        book_factors_df['genre'] = books_df['genre'].values
    else:
        # genre 정보가 없으면 'Unknown' 사용
        book_factors_df['genre'] = ['Unknown' for _ in range(len(books_df))]
    
    user_factors_df = pd.DataFrame()
    for level, indices in enumerate(user_indices):
        user_factors_df[f'level_{level+1}_factor'] = indices.numpy()
    
    user_factors_df['user_id'] = users_df['user_id'].values
    
    # 결과 저장
    book_factors_df.to_csv(f"{data_dir}/book_factors.csv", index=False)
    user_factors_df.to_csv(f"{data_dir}/user_factors.csv", index=False)
    
    # 6. 장르별 잠재 요인 분포 분석 (장르 필드가 있는 경우에만 실행)
    print("\n6. 장르별 잠재 요인 분포 분석 중...")
    
    if len(set(book_factors_df['genre'])) > 1:  # 같은 'Unknown' 값만 있는 경우 제외
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
        print("- genre_factor_distribution.png: 장르별 잠재 요인 분포")
    else:
        print("장르 정보가 부족하여 장르별 분포 분석을 건너뜁니다.")
    
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
        book_vectors_array, user_vectors_array, 
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
    book_vectors_array, user_vectors_array, 
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
    print("\n9. 추천 결과 생성 중...")
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
    # 첫 번째 사용자 ID 선택
    if len(users_df) > 0:
        test_user_id = users_df['user_id'].iloc[0]  # 데이터프레임의 첫 번째 사용자 ID 사용
        test_user_idx = 0  # 첫 번째 사용자의 인덱스
        
        # NumPy 배열을 PyTorch 텐서로 변환
        if isinstance(user_vectors, np.ndarray):
            test_user_vector = torch.tensor(user_vectors[test_user_idx], dtype=torch.float32).unsqueeze(0)
        else:
            test_user_vector = user_vectors[test_user_idx].unsqueeze(0)
            
        if isinstance(user_graph_emb, np.ndarray):
            test_user_graph_emb = torch.tensor(user_graph_emb[test_user_idx], dtype=torch.float32).unsqueeze(0)
        else:
            test_user_graph_emb = user_graph_emb[test_user_idx].unsqueeze(0)
    else:
        print("경고: 사용자 데이터가 없습니다. 추천을 생성할 수 없습니다.")
        return None

    # 모든 책에 대한 점수 계산 (여기서 book_graph_emb 사용)
    scores = []
    for i in range(len(books_df)):
        # NumPy 배열을 PyTorch 텐서로 변환
        if isinstance(book_vectors, np.ndarray):
            book_vector = torch.tensor(book_vectors[i], dtype=torch.float32).unsqueeze(0)
        else:
            book_vector = book_vectors[i].unsqueeze(0)
            
        if isinstance(book_graph_emb, np.ndarray):
            book_graph_emb_i = torch.tensor(book_graph_emb[i], dtype=torch.float32).unsqueeze(0)
        else:
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
        'score': scores
    })
    
    # 만약 title, author, genre 컴럼이 있다면 추가
    if 'title' in books_df.columns:
        recommendations['title'] = books_df['title']
    else:
        # book_id를 임시 제목으로 사용
        recommendations['title'] = [f'Book {id}' for id in books_df['book_id']]
        
    if 'author' in books_df.columns:
        recommendations['author'] = books_df['author']
    else:
        recommendations['author'] = ['Unknown' for _ in range(len(books_df))]
        
    if 'genre' in books_df.columns:
        recommendations['genre'] = books_df['genre']
    else:
        recommendations['genre'] = ['Unknown' for _ in range(len(books_df))]

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
    # 제목이 너무 길면 줄임
    display_titles = [title[:20] + '...' if len(title) > 20 else title for title in top_recommendations['title']]
    plt.bar(display_titles, top_recommendations['score'])
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
    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)
    
    # matplotlib 설정 (Agg 백엔드 사용)
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 이미지 파일로 저장
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Apple SD Gothic Neo"  # 또는 "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
    import seaborn as sns
    
    # 메인 함수 실행
    main()