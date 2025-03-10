import torch
from transformers import AutoTokenizer, AutoModel

class SemanticVectorGenerator:
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
    
    def generate_user_prompts(self, user_profile, user_histories):
        """_summary_

        Args:
            user_profile (_type_): _description_
            user_histories (_type_): _description_
        
        
        Returns:
            llm에 전달할 프롬프트 리스트
        """
        prompts = []
        
        for attrs in item_attributes:
            title = attrs.get("title", "")
            category = attrs.get("category", "")
            
            prompt =f"Introduce {category} {title} and describe its attributes precisely." \
                f"(including but not limited to genre, characters, plot, topic/theme,writingstyle, production quality, etc.)."
            prompts.append(prompt)
            
        return prompts
    
        