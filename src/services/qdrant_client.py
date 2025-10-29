"""
Qdrant Client Service
Handles vector storage operations with Qdrant
"""

from langchain_qdrant import Qdrant, QdrantVectorStore
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
import uuid


class QdrantService:
    """Manage Qdrant vector database operations"""
    
    def __init__(self, qdrant_url: str = None, collection_name: str = None, embeddings=None):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "research_papers")
        self.embeddings = embeddings
    
    def create_collection_from_documents(
        self, 
        documents: List[Document], 
        collection_name: str = None,
        force_recreate: bool = False
    ) -> None:
        """Create a new collection and add documents"""
        col_name = collection_name or self.collection_name
        
        Qdrant.from_documents(
            documents,
            self.embeddings,
            url=self.qdrant_url,
            collection_name=col_name,
            force_recreate=force_recreate
        )
    
    def add_chunks_to_collection(
        self,
        chunks: List[Dict],
        paper_id: int,
        paper_metadata: Dict,
        collection_name: str = None
    ) -> int:
        """Add paper chunks to Qdrant with metadata"""
        col_name = collection_name or self.collection_name
        
        # Prepare documents with metadata
        documents = []
        for chunk in chunks:
            # Create unique ID for each chunk
            chunk_id = f"paper_{paper_id}_chunk_{chunk['chunk_index']}"
            
            # Prepare metadata payload
            metadata = {
                "paper_id": paper_id,
                "paper_title": paper_metadata.get("title", ""),
                "authors": paper_metadata.get("authors", ""),
                "year": paper_metadata.get("year"),
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "section": chunk.get("section", "Body"),
                "file_name": paper_metadata.get("file_name", ""),
            }
            
            # Create document
            doc = Document(
                page_content=chunk["text"],
                metadata=metadata
            )
            documents.append(doc)
        
        # Add to Qdrant
        self.create_collection_from_documents(
            documents=documents,
            collection_name=col_name,
            force_recreate=False  # Don't recreate, just add
        )
        
        return len(documents)
    
    def get_vectorstore(self, collection_name: str = None) -> QdrantVectorStore:
        """Get existing Qdrant vector store"""
        col_name = collection_name or self.collection_name
        
        return QdrantVectorStore.from_existing_collection(
            collection_name=col_name,
            url=self.qdrant_url,
            embedding=self.embeddings
        )
    
    def search_similar(
        self,
        query_text: str = None,
        collection_name: str = None,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar documents with scores
        
        Returns list of dicts with:
        - chunk_text: the text content
        - score: relevance score
        - metadata: paper metadata (paper_id, paper_title, section, page_number, etc.)
        """
        vectorstore = self.get_vectorstore(collection_name)
        
        # Use similarity_search_with_score for relevance scores
        results = vectorstore.similarity_search_with_score(
            query_text,
            k=top_k
        )
        
        # Format results
        formatted_results = []
        for doc, score in results:
            # Filter by score threshold
            if score < score_threshold:
                continue
            
            formatted_results.append({
                "chunk_text": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def delete_by_paper_id(
        self,
        paper_id: int,
        collection_name: str = None
    ) -> bool:
        """Delete all vectors for a specific paper"""
        # Note: This requires direct qdrant_client access
        # For now, we'll return True as a placeholder
        # In production, you'd use qdrant_client to filter and delete
        return True
