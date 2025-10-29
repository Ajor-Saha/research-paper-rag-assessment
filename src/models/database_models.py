from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON, Boolean, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database import Base


class Paper(Base):
    """Research paper metadata"""
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    authors = Column(String, nullable=False)
    year = Column(Integer, nullable=True, index=True)
    abstract = Column(Text, nullable=True)
    file_path = Column(String, nullable=False, unique=True)
    file_name = Column(String, nullable=False, index=True)
    file_size = Column(Integer, nullable=True)  # in bytes
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, default=0)  # Track chunk count from Qdrant
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True)
    qdrant_collection_name = Column(String, nullable=True)  # Collection name in Qdrant (e.g., "research_papers")
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)  # Store error message if processing fails
    is_deleted = Column(Boolean, default=False)
    
    # Relationships
    query_papers = relationship("QueryPaper", back_populates="paper", cascade="all, delete-orphan")


class QueryHistory(Base):
    """Store query history and analytics"""
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False, index=True)
    answer = Column(Text, nullable=False)
    response_time = Column(Float, nullable=False)  # in seconds
    confidence_score = Column(Float, nullable=True)
    top_k = Column(Integer, default=5)  # Number of chunks retrieved
    sources_used = Column(JSON, nullable=True)  # List of paper IDs/names used
    citations = Column(JSON, nullable=True)  # Citations data
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    user_rating = Column(Integer, nullable=True)  # Optional 1-5 rating
    
    # Relationships
    query_papers = relationship("QueryPaper", back_populates="query", cascade="all, delete-orphan")


class QueryPaper(Base):
    """Many-to-many relationship between queries and papers"""
    __tablename__ = "query_papers"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    relevance_score = Column(Float, nullable=True)  # How relevant this paper was to the query
    chunks_used = Column(Integer, default=0)  # Number of chunks from this paper used
    
    # Relationships
    query = relationship("QueryHistory", back_populates="query_papers")
    paper = relationship("Paper", back_populates="query_papers")
    
    __table_args__ = (
        Index('idx_query_paper', 'query_id', 'paper_id'),
    )
