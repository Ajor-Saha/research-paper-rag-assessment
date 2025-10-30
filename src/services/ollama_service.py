"""
Ollama Service
Handles LLM operations using Ollama
"""

from langchain_ollama import OllamaLLM
from typing import Dict, Optional
import os
import json


class OllamaService:
    """Manage LLM operations with Ollama"""
    
    def __init__(self, model: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2:2b")
        
        # Auto-detect environment: use 'ollama' in Docker, 'localhost' locally
        base_url = os.getenv("OLLAMA_BASE_URL")
        if not base_url:
            # Try to detect if we're in Docker by checking if 'ollama' hostname resolves
            import socket
            try:
                socket.gethostbyname('ollama')
                base_url = "http://ollama:11434"  # In Docker
            except socket.error:
                base_url = "http://localhost:11434"  # Local
        
        print(f"🔗 Connecting to Ollama at: {base_url}")
        self.llm = OllamaLLM(model=self.model, base_url=base_url, temperature=0.3)
    
    def extract_paper_metadata(self, text: str) -> Dict:
        """Extract metadata from paper text using LLM"""
        
        # Take first 2000 characters for metadata extraction
        text_excerpt = text[:2000]
        
        prompt = f"""You are a research paper metadata extractor. Extract the following information from the given text:
- Title: The paper title
- Authors: All author names (comma-separated)
- Year: Publication year
- Abstract: The abstract text (if present)

Text:
{text_excerpt}

Return ONLY a valid JSON object with keys: title, authors, year, abstract
If any field is not found, use null (JSON null, not the string "null").
Example: {{"title": "Paper Title", "authors": "John Doe, Jane Smith", "year": 2023, "abstract": "This paper..."}}

JSON:"""

        try:
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            json_str = response.strip()
            
            # Try to find JSON in the response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            metadata = json.loads(json_str)
            
            # Clean up values - convert string "null" to None
            for key in ["title", "authors", "abstract"]:
                value = metadata.get(key)
                if value in ["null", "None", "N/A", "n/a", "undefined", ""]:
                    metadata[key] = None
            
            # Handle year specially
            year = metadata.get("year")
            if year in ["null", "None", None, "", "N/A"]:
                metadata["year"] = None
            elif isinstance(year, str):
                try:
                    # Extract digits from string
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', year)
                    if year_match:
                        metadata["year"] = int(year_match.group())
                    else:
                        metadata["year"] = None
                except:
                    metadata["year"] = None
            
            return metadata
            
        except Exception as e:
            print(f"Error extracting metadata with LLM: {e}")
            # Fallback - return None values
            return {
                "title": None,
                "authors": None,
                "year": None,
                "abstract": None
            }
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on context"""
        prompt = f"""You are a helpful assistant answering questions based on research papers.

Context:
{context}

Question: {question}

Provide a clear and concise answer based on the context. If the answer is not in the context, say so.

Answer:"""
        
        return self.llm.invoke(prompt)
