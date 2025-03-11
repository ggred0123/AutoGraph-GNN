import torch
from transformers import AutoTokenizer, AutoModel

class BookSemanticVectorGenerator:
    def __init__(self, llm_model_name:"gpt-2"):
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

        
            