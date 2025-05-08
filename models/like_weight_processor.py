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
        
    def apply_likes_weights(self, book_vectors, user_vectors, likes_df):
        """
        좋아요 데이터를 기존 임베딩에 가중치로 적용
        
        Args:
            book_vectors (numpy.ndarray or torch.Tensor): 원본 책 임베딩 벡터
            user_vectors (numpy.ndarray or torch.Tensor): 원본 사용자 임베딩 벡터
            likes_df (pd.DataFrame): 좋아요 데이터 (user_id, book_id 컬럼 포함)
            
        Returns:
            tuple: (가중치 적용된 책 임베딩, 가중치 적용된 사용자 임베딩)
        """
        # 입력 배열 형식 확인
        is_torch_book = isinstance(book_vectors, torch.Tensor)
        is_torch_user = isinstance(user_vectors, torch.Tensor)
        
        # NumPy 배열이면 복사, PyTorch 텐서면 clone() 사용
        if is_torch_book:
            weighted_book_vectors = book_vectors.clone()
        else:
            weighted_book_vectors = book_vectors.copy()
            
        if is_torch_user:
            weighted_user_vectors = user_vectors.clone()
        else:
            weighted_user_vectors = user_vectors.copy()
        
        if len(likes_df) == 0:
            # 좋아요 데이터가 없는 경우 원본 배열 그대로 반환
            return weighted_book_vectors, weighted_user_vectors
            
        # 사용자별 좋아요 그룹화
        user_likes = likes_df.groupby('user_id')['book_id'].apply(list).to_dict()
        
        # 책별 좋아요 그룹화
        book_likes = likes_df.groupby('book_id')['user_id'].apply(list).to_dict()
        
        # 사용자와 책 ID 목록
        user_ids = list(user_likes.keys())
        book_ids = list(likes_df['book_id'].unique())
        
        # books_df에서 book_ids 목록 얻기
        global_book_ids = None
        try:
            if 'book_ids' in globals():
                global_book_ids = globals()['book_ids']
        except:
            pass
        
        # 사용자 임베딩에 좋아요 정보 반영
        for i, user_id in enumerate(user_ids):
            try:
                # 0부터 시작하는 인덱스 사용
                user_idx = i
                if user_idx < 0 or user_idx >= len(weighted_user_vectors):
                    continue  # 인덱스 범위 확인
                
                # 해당 사용자가 좋아하는 책들의 임베딩 평균 계산
                liked_books = user_likes[user_id]
                liked_book_vectors = []
                
                for book_id in liked_books:
                    try:
                        # 책의 인덱스 찾기
                        book_idx = None
                        if global_book_ids is not None and book_id in global_book_ids:
                            book_idx = global_book_ids.index(book_id)
                        else:
                            # 임시 방편: book_id가 정수이고 인덱스 범위 내에 있다면 그대로 사용
                            if isinstance(book_id, int) and 0 <= book_id < len(weighted_book_vectors):
                                book_idx = book_id
                            else:
                                # 아니면 book_ids 리스트에서 찾기
                                book_idx = book_ids.index(book_id) if book_id in book_ids else None
                                
                        if book_idx is not None and 0 <= book_idx < len(weighted_book_vectors):
                            liked_book_vectors.append(weighted_book_vectors[book_idx])
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"책 인덱스 찾기 오류 (book_id: {book_id}): {e}")
                        continue
                
                if liked_book_vectors:
                    # NumPy 배열이나 PyTorch 텐서에 따라 다르게 처리
                    if is_torch_book:
                        liked_books_avg = torch.stack(liked_book_vectors).mean(dim=0)
                    else:
                        liked_books_avg = np.mean(np.array(liked_book_vectors), axis=0)
                    
                    # 가중 평균: (1-alpha) * 원래 벡터 + alpha * 좋아요 책 평균
                    weighted_user_vectors[user_idx] = (1 - self.alpha) * weighted_user_vectors[user_idx] + self.alpha * liked_books_avg
            except Exception as e:
                print(f"사용자 {user_id} 처리 중 오류: {e}")
        
        # 책 임베딩에 좋아요 정보 반영
        for book_id in book_likes.keys():
            try:
                # 책의 인덱스 찾기
                book_idx = None
                if global_book_ids is not None and book_id in global_book_ids:
                    book_idx = global_book_ids.index(book_id)
                else:
                    # 임시 방편: book_id가 정수이고 인덱스 범위 내에 있다면 그대로 사용
                    if isinstance(book_id, int) and 0 <= book_id < len(weighted_book_vectors):
                        book_idx = book_id
                    else:
                        # 아니면 book_ids 리스트에서 찾기
                        book_idx = book_ids.index(book_id) if book_id in book_ids else None
                
                if book_idx is None or book_idx < 0 or book_idx >= len(weighted_book_vectors):
                    continue  # 인덱스 범위 확인
                
                # 이 책을 좋아하는 사용자들의 임베딩 평균 계산
                liked_by_users = book_likes[book_id]
                liking_user_vectors = []
                
                for j, user_id in enumerate(liked_by_users):
                    try:
                        # 사용자의 인덱스 찾기
                        user_idx = None
                        if user_id in user_ids:
                            user_idx = user_ids.index(user_id)
                        
                        if user_idx is not None and 0 <= user_idx < len(weighted_user_vectors):
                            liking_user_vectors.append(weighted_user_vectors[user_idx])
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"사용자 인덱스 찾기 오류 (user_id: {user_id}): {e}")
                        continue
                
                if liking_user_vectors:
                    # NumPy 배열이나 PyTorch 텐서에 따라 다르게 처리
                    if is_torch_user:
                        liking_users_avg = torch.stack(liking_user_vectors).mean(dim=0)
                    else:
                        liking_users_avg = np.mean(np.array(liking_user_vectors), axis=0)
                    
                    # 가중 평균: (1-alpha) * 원래 벡터 + alpha * 좋아요 사용자 평균
                    weighted_book_vectors[book_idx] = (1 - self.alpha) * weighted_book_vectors[book_idx] + self.alpha * liking_users_avg
            except Exception as e:
                print(f"책 {book_id} 처리 중 오류: {e}")
        
        return weighted_book_vectors, weighted_user_vectors