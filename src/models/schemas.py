from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional, Dict, Any


# Paper Schemas
class PaperBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    authors: str = Field(..., min_length=1)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    abstract: Optional[str] = None


class PaperCreate(PaperBase):
    file_path: str
    file_name: str
    file_size: Optional[int] = None
    total_pages: int = Field(..., ge=1)
    total_chunks: int = 0
    qdrant_collection_name: Optional[str] = None
    processing_status: str = "pending"


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = Field(None, ge=1900, le=2100)
    abstract: Optional[str] = None
    processing_status: Optional[str] = None


class PaperResponse(PaperBase):
    id: int
    file_name: str
    total_pages: int
    uploaded_at: datetime
    processing_status: str
    
    class Config:
        from_attributes = True


class PaperDetail(PaperResponse):
    file_path: str
    file_size: Optional[int] = None
    total_chunks: int
    qdrant_collection_name: Optional[str] = None
    query_count: Optional[int] = 0


class PaperListResponse(BaseModel):
    papers: List[PaperResponse]
    total: int
    page: int
    page_size: int


class PaperStats(BaseModel):
    id: int
    title: str
    authors: str
    total_pages: int
    total_chunks: int
    total_queries: int
    avg_response_time: Optional[float] = 0.0
    avg_confidence: Optional[float] = 0.0
    most_common_sections_queried: Optional[List[str]] = []
    upload_date: datetime


# Query Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    paper_ids: Optional[List[int]] = Field(default=None, description="Optional: limit search to specific papers")
    
    @validator('question')
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class Citation(BaseModel):
    paper_id: int
    paper_title: str
    section: Optional[str] = None
    page: int
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    chunk_text: Optional[str] = None  # Preview of the relevant text


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    sources_used: List[str]  # List of paper titles/filenames
    confidence: float = Field(..., ge=0.0, le=1.0)
    response_time: float
    query_id: Optional[int] = None


class QueryHistoryResponse(BaseModel):
    id: int
    query_text: str
    answer: str
    response_time: float
    confidence_score: Optional[float] = None
    sources_used: Optional[List[Any]] = []
    citations: Optional[List[Dict]] = []
    created_at: datetime
    user_rating: Optional[int] = None
    papers_referenced: Optional[List[str]] = []
    
    class Config:
        from_attributes = True


class QueryHistoryListResponse(BaseModel):
    queries: List[QueryHistoryResponse]
    total: int
    page: int
    page_size: int


class QueryRating(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    
    @validator('rating')
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


# Analytics Schemas
class PopularQuery(BaseModel):
    query_text: str
    count: int
    avg_confidence: Optional[float] = 0.0
    avg_response_time: Optional[float] = 0.0


class PopularPaper(BaseModel):
    paper_id: int
    paper_title: str
    query_count: int
    avg_relevance: Optional[float] = 0.0


class AnalyticsSummary(BaseModel):
    total_papers: int
    total_queries: int
    total_chunks: int
    avg_response_time: float
    avg_confidence: float


class AnalyticsResponse(BaseModel):
    summary: AnalyticsSummary
    popular_queries: List[PopularQuery]
    popular_papers: List[PopularPaper]
    queries_over_time: Optional[Dict[str, int]] = {}  # Date: count


class TopicAnalytics(BaseModel):
    topic: str
    query_count: int
    avg_confidence: float
    related_papers: List[str]


# Upload Response
class UploadResponse(BaseModel):
    message: str
    paper_id: int
    title: str
    authors: str
    total_pages: int
    chunks_created: int
    processing_time: float
    status: str = "completed"


class DeleteResponse(BaseModel):
    message: str
    paper_id: int
    chunks_deleted: int
    vectors_deleted: bool


# Error Response
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Success Response
class SuccessResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
