class Config:
    def __init__(self):
        
        self.llm_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        self.semantic_vector_dim = 385
        self.hidden_dim = 128
        
        self.user_codebook_size = 32
        self.item_codebook_size = 64
        
        self.num_codebooks = 2
        
        self.output_dim = 1
        
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.weight_decay = 1e-5
        self.early_stop_patience = 5
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
def get_config():
    
    return Config()