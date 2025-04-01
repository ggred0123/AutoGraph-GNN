from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from utils.db_connecter import get_db_connection


get_db_connection()
# FastAPI 앱 초기화
app = FastAPI(title="GNN Book Recommendation API")

class RecommendRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    filter_read: bool = True

class UserPreferenceUpdate(BaseModel):
    user_id: int
    book_ids: List[int]
    ratings: Optional[List[float]] = None

# 루트 경로
@app.get("/")
async def root():
    """API 루트 경로"""
    return {
        "message": "GNN 기반 도서 추천 API",
        "version": "0.1.0",
        "status": "서버가 실행 중이지만 모델과 데이터는 아직 로드되지 않았습니다.",
        "docs_url": "/docs"
    }

@app.get("/recommend/{user_id}")
async def recommend_books(
    user_id: int, 
    num_recommendations: int = Query(10, ge=1, le=50),
    filter_read: bool = True
):
    """특정 사용자에게 책을 추천합니다."""
    # 임시 응답 데이터
    return {
        "user_id": user_id,
        "status": "준비 중",
        "message": "추천 시스템이 아직 초기화되지 않았습니다.",
        "recommendations": []
    }

@app.get("/books")
async def get_books(genre: Optional[str] = None, limit: int = Query(50, ge=1, le=100)):
    """모든 책 목록 또는 장르별 책 목록을 반환합니다."""
    return {
        "status": "준비 중",
        "message": "도서 데이터베이스가 아직 로드되지 않았습니다.",
        "books": []
    }

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """특정 사용자 정보를 반환합니다."""
    return {
        "status": "준비 중",
        "message": "사용자 데이터베이스가 아직 로드되지 않았습니다.",
        "user_id": user_id
    }

@app.post("/update-preferences")
async def update_user_preferences(preferences: UserPreferenceUpdate):
    """사용자의 책 선호도를 업데이트합니다."""
    return {
        "status": "준비 중",
        "message": "사용자 선호도 업데이트 시스템이 아직 초기화되지 않았습니다.",
        "user_id": preferences.user_id
    }

@app.get("/health")
async def health_check():
    """서버 상태를 확인합니다."""
    return {
        "status": "server_only",
        "message": "서버는 실행 중이지만 모델과 데이터는 로드되지 않았습니다."
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)