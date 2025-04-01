import torch
import pandas as pd
import numpy as np

class LikesWeightProcessor:
    """
    유저의 좋아요 데이터를 임베딩 가중치로 활용하는 프로세서
    """
    def __init__(self, alpha=0.3):
        """
        생성자
        Args:
            alpha (float): 좋아요 데이터의 가중치 계수 (0~1 사이)
                          값이 클수록 좋아요의 영향이 커짐
        """
        self.alpha = alpha
        
    def apply_likes_weights(self, user_vectors, book_vectors, likes_df):
        """
        좋아요 데이터를 기존 임베딩에 가중치로 적용
        
        Args:
            user_vectors (torch.Tensor): 원본 사용자 임베딩 벡터
            book_vectors (torch.Tensor): 원본 책 임베딩 벡터
            likes_df (pd.DataFrame): 좋아요 데이터 (user_id, book_id 컬럼 포함)
            
        Returns:
            tuple: (가중치 적용된 사용자 임베딩, 가중치 적용된 책 임베딩)
        """
        # 임베딩 복사 (원본 변경 방지)
        weighted_user_vectors = user_vectors.clone()
        weighted_book_vectors = book_vectors.clone()
        
        # 사용자별 좋아요 그룹화
        user_likes = likes_df.groupby('user_id')['book_id'].apply(list).to_dict()
        
        # 책별 좋아요 그룹화
        book_likes = likes_df.groupby('book_id')['user_id'].apply(list).to_dict()
        
        # 사용자 임베딩에 좋아요 정보 반영
        for user_id, liked_books in user_likes.items():
            try:
                user_idx = user_id - 1  # 가정: user_id가 1부터 시작하는 경우
                if user_idx < 0 or user_idx >= len(weighted_user_vectors):
                    continue  # 인덱스 범위 확인
                
                # 해당 사용자가 좋아하는 책들의 임베딩 평균 계산
                liked_book_vectors = []
                for book_id in liked_books:
                    book_idx = book_id - 1  # 가정: book_id가 1부터 시작하는 경우
                    if 0 <= book_idx < len(weighted_book_vectors):
                        liked_book_vectors.append(weighted_book_vectors[book_idx])
                
                if liked_book_vectors:
                    liked_books_avg = torch.stack(liked_book_vectors).mean(dim=0)
                    # 가중 평균: (1-alpha) * 원래 벡터 + alpha * 좋아요 책 평균
                    weighted_user_vectors[user_idx] = (1 - self.alpha) * weighted_user_vectors[user_idx] + self.alpha * liked_books_avg
            except Exception as e:
                print(f"사용자 {user_id} 처리 중 오류: {e}")
        
        # 책 임베딩에 좋아요 정보 반영
        for book_id, liked_by_users in book_likes.items():
            try:
                book_idx = book_id - 1  # 가정: book_id가 1부터 시작하는 경우
                if book_idx < 0 or book_idx >= len(weighted_book_vectors):
                    continue  # 인덱스 범위 확인
                
                # 이 책을 좋아하는 사용자들의 임베딩 평균 계산
                liking_user_vectors = []
                for user_id in liked_by_users:
                    user_idx = user_id - 1  # 가정: user_id가 1부터 시작하는 경우
                    if 0 <= user_idx < len(weighted_user_vectors):
                        liking_user_vectors.append(weighted_user_vectors[user_idx])
                
                if liking_user_vectors:
                    liking_users_avg = torch.stack(liking_user_vectors).mean(dim=0)
                    # 가중 평균: (1-alpha) * 원래 벡터 + alpha * 좋아요 사용자 평균
                    weighted_book_vectors[book_idx] = (1 - self.alpha) * weighted_book_vectors[book_idx] + self.alpha * liking_users_avg
            except Exception as e:
                print(f"책 {book_id} 처리 중 오류: {e}")
        
        return weighted_user_vectors, weighted_book_vectors