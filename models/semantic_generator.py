import torch
from transformers import AutoTokenizer, AutoModel

class BookSemanticVectorGenerator:
    def __init__(self, llm_model_name:"sentence-transformers/all-MiniLM-L6-v2"):
        """_summary_

        Args:
            llm_model_name (gpt): 사전 훈련된 언어 모델 이름
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModel.from_pretrained(llm_model_name)
        self.model.eval()
        
    def generate_vectors(self, descriptions, max_length = 128):
        """_summary_

        Args:
            descriptions (_type_): _description_
            max_length (int, optional): _description_. Defaults to 128.
        """
        
        inputs = self.tokenizer(descriptions, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            vectors = outputs.last_hidden_state[:, 0, :]
            
        return vectors
    
    def generate_user_prompts(self, user_profiles, user_histories):
        prompts = []
        
        for i, profile in enumerate(user_profiles):
            user_id = profile.get("user_id", i)
            age = profile.get("age", "unknown age")
            gender = profile.get("gender", "unknown gender")
            
            # 관심 장르가 JSON 문자열로 저장되어 있으면 파싱
            favorite_genres = profile.get("favorite_genres", "[]")
            if isinstance(favorite_genres, str) and favorite_genres.startswith("["):
                import json
                favorite_genres = json.loads(favorite_genres)
            else:
                favorite_genres = []
            
            # 상호작용 기록을 문자열로 변환
            history = user_histories.get(user_id, [])
            history_str = ", ".join([f"{i+1}. {item}" for i, item in enumerate(history[:3])])
            
            prompt = f"Given a {gender} user who is aged {age} and interested in {', '.join(favorite_genres)}, "\
                    f"this user's reading history is listed below:\n{history_str}. "\
                    f"Analyze the user's preferences (consider factors like genre, author, topics, "\
                    f"writing style, etc.). Provide clear explanations based on "\
                    f"relevant details from the user's reading history."
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_item_prompts(self, item_attributes):
        prompts = []
        
        for attrs in item_attributes.to_dict('records'):
            title = attrs.get("title", "unknown title")
            author = attrs.get("author", "unknown author")
            genre = attrs.get("genre", "")
            summary = attrs.get("summary", "")
            
            prompt = f"Book title: {title}\nAuthor: {author}\nGenre: {genre}\n"\
                    f"Summary: {summary}\n\n"\
                    f"Analyze this book's characteristics, themes, writing style, and "\
                    f"potential audience. Consider what makes it unique or similar to other books."
            
            prompts.append(prompt)
        
        return prompts


    
    def generate_book_embeddings(self, item_attributes, batch_size=16, combine_method='concat'):
        """
        책 속성에 기반한 텍스트 프롬프트를 생성하고, 이를 LLM을 통해 임베딩 벡터로 변환합니다.
        
        Args:
            item_attributes (pd.DataFrame): 책의 속성을 담은 DataFrame (컬럼: title, author, genre, summary 등)
            batch_size (int, optional): 배치 처리 크기. Defaults to 16.
            combine_method (str, optional): 'concat'이면 각 책별 임베딩을 그대로 반환,
                                            'mean'이면 배치 내 임베딩의 평균을 반환합니다.
                                            (일반적으로 'concat' 사용)
        
        Returns:
            torch.Tensor: (책 수, 임베딩 차원) 크기의 텐서
        """
        prompts = self.generate_item_prompts(item_attributes)
        all_vectors = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_vectors = self.generate_vectors(batch_prompts)
            all_vectors.append(batch_vectors)
        embeddings = torch.cat(all_vectors, dim=0)
        
        if combine_method == 'concat':
            return embeddings
        elif combine_method == 'mean':
            return embeddings.mean(dim=0, keepdim=True)
        else:
            raise ValueError("Unknown combine_method. Choose 'concat' or 'mean'.")

    def generate_user_embeddings(self, user_profiles, interactions_df, books_df, method='weighted', batch_size=16):
        """
        사용자 프로필, 상호작용 기록, 책 속성을 활용해 사용자 프롬프트를 생성하고, 
        LLM을 통해 사용자 임베딩을 추출합니다.
        
        Args:
            user_profiles (pd.DataFrame): 사용자 정보를 담은 DataFrame (예: user_id, age, gender, favorite_genres 등)
            interactions_df (pd.DataFrame): 상호작용 기록 DataFrame (컬럼: user_id, book_id)
            books_df (pd.DataFrame): 책 정보를 담은 DataFrame (컬럼: book_id, title, 등)
            method (str, optional): 임베딩 생성 방식 (현재는 'weighted' 처리와 동일하게 배치 처리). Defaults to 'weighted'.
            batch_size (int, optional): 배치 처리 크기. Defaults to 16.
        
        Returns:
            torch.Tensor: (사용자 수, 임베딩 차원) 크기의 텐서
        """
        # books_df에서 book_id -> title 매핑 생성
        book_map = books_df.set_index("book_id")["title"].to_dict()
        user_histories = {}
        for _, row in interactions_df.iterrows():
            uid = row["user_id"]
            bid = row["book_id"]
            title = book_map.get(bid, "")
            if uid in user_histories:
                user_histories[uid].append(title)
            else:
                user_histories[uid] = [title]
        
        # DataFrame을 dict 목록으로 변환
        user_profiles_list = user_profiles.to_dict('records')
        
        # 사용자 프롬프트 생성
        prompts = self.generate_user_prompts(user_profiles_list, user_histories)
        
        all_vectors = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_vectors = self.generate_vectors(batch_prompts)
            all_vectors.append(batch_vectors)
        embeddings = torch.cat(all_vectors, dim=0)
        
        # method 인자에 따라 다른 처리가 필요하다면 여기서 추가 구현 가능 (현재는 기본 배치처리 방식)
        return embeddings    
    
            