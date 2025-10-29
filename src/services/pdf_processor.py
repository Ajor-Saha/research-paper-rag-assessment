"""
PDF Processing Service
Handles PDF loading, text extraction, and intelligent chunking
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional
import re
import os


class PDFProcessor:
    """Handle PDF processing and text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, llm_service=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_service = llm_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and extract documents with page info"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def extract_metadata(self, documents: List[Document]) -> Dict:
        """Extract paper metadata from documents using LLM"""
        if not documents:
            return {}
        
        # Get full text from first few pages for metadata extraction
        full_text = "\n".join([doc.page_content for doc in documents[:3]])
        
        # Use LLM for extraction if available
        if self.llm_service:
            try:
                metadata = self.llm_service.extract_paper_metadata(full_text)
                metadata["total_pages"] = len(documents)
                return metadata
            except Exception as e:
                print(f"LLM extraction failed, using fallback: {e}")
        
        # Fallback to manual extraction
        return {
            "title": self._extract_title(full_text),
            "authors": self._extract_authors(full_text),
            "year": self._extract_year(full_text),
            "abstract": self._extract_abstract(full_text),
            "total_pages": len(documents)
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title from text"""
        # Look for title in first few lines
        lines = text.split('\n')[:10]
        
        # Find longest non-empty line (usually the title)
        title = ""
        for line in lines:
            line = line.strip()
            if len(line) > len(title) and len(line) < 200:
                # Skip lines that look like headers/footers
                if not re.match(r'^\d+$|^page \d+', line.lower()):
                    title = line
        
        return title or "Unknown Title"
    
    def _extract_authors(self, text: str) -> str:
        """Extract authors from text"""
        # Look for common author patterns
        author_patterns = [
            r'(?:by|authors?:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)+)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                return match.group(1)
        
        return "Unknown Authors"
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        # Look for 4-digit year (typically 19xx or 20xx)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        matches = re.findall(year_pattern, text[:2000])
        
        if matches:
            # Return most recent reasonable year
            years = [int(y) for y in matches if 1900 <= int(y) <= 2030]
            return max(years) if years else None
        
        return None
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from text"""
        # Look for abstract section
        abstract_pattern = r'(?:abstract|summary)[:\s]+(.*?)(?:\n\n|\n[A-Z]|\nintroduction)'
        match = re.search(abstract_pattern, text.lower(), re.DOTALL | re.IGNORECASE)
        
        if match:
            abstract = match.group(1).strip()
            # Limit length
            return abstract[:1000] if len(abstract) > 1000 else abstract
        
        return None
    
    def detect_section(self, text: str, page_num: int) -> str:
        """Detect which section of the paper this text belongs to"""
        text_lower = text.lower()[:200]  # Check first 200 chars
        
        # Common section headers
        if any(keyword in text_lower for keyword in ['abstract', 'summary']):
            return "Abstract"
        elif any(keyword in text_lower for keyword in ['introduction', 'background']):
            return "Introduction"
        elif any(keyword in text_lower for keyword in ['method', 'approach', 'technique']):
            return "Methods"
        elif any(keyword in text_lower for keyword in ['result', 'experiment', 'evaluation']):
            return "Results"
        elif any(keyword in text_lower for keyword in ['discussion']):
            return "Discussion"
        elif any(keyword in text_lower for keyword in ['conclusion', 'future work']):
            return "Conclusion"
        elif any(keyword in text_lower for keyword in ['reference', 'bibliography']):
            return "References"
        
        # Default based on page number
        if page_num == 1:
            return "Abstract"
        elif page_num <= 3:
            return "Introduction"
        else:
            return "Body"
    
    def split_into_chunks(self, documents: List[Document]) -> List[Dict]:
        """Split documents into chunks with metadata"""
        chunks = []
        
        for doc_idx, doc in enumerate(documents):
            page_num = doc.metadata.get('page', doc_idx + 1)
            section = self.detect_section(doc.page_content, page_num)
            
            # Split this document into chunks
            doc_chunks = self.text_splitter.split_text(doc.page_content)
            
            for chunk_idx, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "section": section,
                    "chunk_index": len(chunks)  # Global chunk index
                })
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Complete PDF processing pipeline"""
        # Load PDF
        documents = self.load_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(documents)
        
        # Split into chunks
        chunks = self.split_into_chunks(documents)
        
        return {
            "metadata": metadata,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
