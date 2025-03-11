# MovieLensDataset 클래스를 BookDataset 클래스로 변경
class BookDataset:
    """
    도서 데이터셋 로드 및 처리 클래스
    """
    def __init__(self, path="./data/", min_interactions=5):
        self.path = path
        self.min_interactions = min_interactions
        self.books = {}
        self.users = {}
        self.user_histories = {}
        self.interactions = []
        
        self.load_data()
        
    def load_data(self):
        """도서 데이터 로드 및 전처리"""
        try:
            # 파일에서 데이터 로드
            self.books = pd.read_csv(f"{self.path}/books.csv")
            self.users = pd.read_csv(f"{self.path}/users.csv")
            interactions_df = pd.read_csv(f"{self.path}/interactions.csv")
            
            # 사용자별 읽은 책 기록 구성
            self.user_histories = {}
            for user_id in self.users['user_id'].unique():
                user_books = interactions_df[interactions_df['user_id'] == user_id]
                self.user_histories[user_id] = self.books.loc[user_books['book_id'] - 1, 'title'].tolist()
            
            # 상호작용 데이터 준비
            self.interactions = list(zip(interactions_df['user_id'], interactions_df['book_id']))
            
        except FileNotFoundError:
            print("데이터 파일을 찾을 수 없습니다. 가상 데이터를 생성합니다.")
            from book_dataset import generate_book_dataset, generate_user_dataset
            
            self.books = generate_book_dataset(num_books=100)
            self.users, interactions_df = generate_user_dataset(num_users=50, num_books=100)
            
            # 사용자별 읽은 책 기록 구성
            self.user_histories = {}
            for user_id in self.users['user_id'].unique():
                user_books = interactions_df[interactions_df['user_id'] == user_id]
                self.user_histories[user_id] = self.books.loc[user_books['book_id'] - 1, 'title'].tolist()
            
            # 상호작용 데이터 준비
            self.interactions = list(zip(interactions_df['user_id'], interactions_df['book_id']))
    
    # 나머지 메서드는 동일하게 유지...
    def get_user_profiles(self):
        return self.users
        
    def get_item_attributes(self):
        return self.books
        
    def get_user_histories(self):
        return self.user_histories
        
    def get_interactions(self):
        return self.interactions