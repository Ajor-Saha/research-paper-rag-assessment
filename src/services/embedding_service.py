"""
Embedding Service
Handles text embeddings generation using sentence-transformers
"""

from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os


class EmbeddingService:
    """Generate embeddings for text using HuggingFace models"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        test_embedding = self.embed_text("test")
        return len(test_embedding)
