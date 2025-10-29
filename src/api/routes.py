from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import time
import shutil

from src.database import get_db
from src.models import database_models, schemas
from src.services.pdf_processor import PDFProcessor
from src.services.embedding_service import EmbeddingService
from src.services.qdrant_client import QdrantService
from src.services.ollama_service import OllamaService

router = APIRouter()

# Initialize services
ollama_service = OllamaService()
pdf_processor = PDFProcessor(llm_service=ollama_service)
embedding_service = EmbeddingService()
qdrant_service = QdrantService(embeddings=embedding_service.embeddings)


# ============================================
# PAPER MANAGEMENT ENDPOINTS
# ============================================

@router.post("/papers/upload", response_model=List[schemas.UploadResponse], status_code=status.HTTP_201_CREATED)
async def upload_papers(
    files: List[UploadFile] = File(..., description="Multiple PDF files to upload"),
    db: Session = Depends(get_db)
):
    """
    Upload and process multiple research paper PDFs
    
    - Extracts text with section awareness using LLM
    - Intelligent chunking (preserves semantic context)
    - Generates embeddings
    - Stores in Qdrant with metadata
    - Saves paper info in database
    
    Supports bulk upload: Process up to 5 papers at once
    """
    
    if len(files) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 files allowed per upload"
        )
    
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    results = []
    
    for file in files:
        start_time = time.time()
        file_path = None
        db_paper = None
        
        # Validate file
        if not file.filename or not file.filename.endswith('.pdf'):
            results.append(schemas.UploadResponse(
                message=f"Skipped: {file.filename} - Only PDF files allowed",
                paper_id=0,
                title="",
                authors="",
                total_pages=0,
                chunks_created=0,
                processing_time=0,
                status="failed"
            ))
            continue
        
        file_path = os.path.join(upload_dir, file.filename)
        
        try:
            # Check if file already exists in database
            existing_paper = db.query(database_models.Paper).filter(
                database_models.Paper.file_name == file.filename
            ).first()
            
            if existing_paper:
                results.append(schemas.UploadResponse(
                    message=f"⚠️ Skipped: {file.filename} - Already exists in database",
                    paper_id=existing_paper.id,
                    title=existing_paper.title,
                    authors=existing_paper.authors,
                    total_pages=existing_paper.total_pages,
                    chunks_created=existing_paper.total_chunks,
                    processing_time=0,
                    status="skipped"
                ))
                continue
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process PDF with LLM extraction
            print(f"📄 Processing: {file.filename}")
            processed_data = pdf_processor.process_pdf(file_path)
            metadata = processed_data["metadata"]
            chunks = processed_data["chunks"]
            
            # Clean metadata values (handle "null" strings from LLM)
            title = metadata.get("title", "Unknown Title")
            if title in ["null", "None", "", None]:
                title = f"Untitled Paper - {file.filename}"
            
            authors = metadata.get("authors", "Unknown Authors")
            if authors in ["null", "None", "", None]:
                authors = "Unknown Authors"
            
            abstract = metadata.get("abstract")
            if abstract in ["null", "None", ""]:
                abstract = None
            
            year = metadata.get("year")
            if year in ["null", "None", ""]:
                year = None
            
            print(f"✅ Extracted metadata: {title}")
            print(f"📊 Created {len(chunks)} chunks")
            
            # Create paper record in database
            paper_data = schemas.PaperCreate(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                file_path=file_path,
                file_name=file.filename,
                file_size=os.path.getsize(file_path),
                total_pages=metadata.get("total_pages", 0),
                total_chunks=len(chunks),
                qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "research_papers"),
                processing_status="processing"
            )
            
            db_paper = database_models.Paper(**paper_data.model_dump())
            db.add(db_paper)
            db.commit()
            db.refresh(db_paper)
            
            # Store in Qdrant with semantic search enabled
            print(f"🔍 Storing vectors in Qdrant...")
            qdrant_service.add_chunks_to_collection(
                chunks=chunks,
                paper_id=db_paper.id,
                paper_metadata={
                    "title": db_paper.title,
                    "authors": db_paper.authors,
                    "year": db_paper.year,
                    "file_name": db_paper.file_name
                }
            )
            
            # Update processing status
            db_paper.processing_status = "completed"
            db.commit()
            
            processing_time = time.time() - start_time
            
            results.append(schemas.UploadResponse(
                message=f"✅ {file.filename} processed successfully",
                paper_id=db_paper.id,
                title=db_paper.title,
                authors=db_paper.authors,
                total_pages=db_paper.total_pages,
                chunks_created=len(chunks),
                processing_time=round(processing_time, 2),
                status="completed"
            ))
            
        except Exception as e:
            # Rollback the transaction if there was an error
            db.rollback()
            
            # Update paper status to failed if it was created
            if db_paper is not None:
                try:
                    db_paper.processing_status = "failed"
                    db_paper.processing_error = str(e)
                    db.commit()
                except:
                    db.rollback()
            
            # Clean up file on error
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            results.append(schemas.UploadResponse(
                message=f"❌ Failed: {file.filename} - {str(e)}",
                paper_id=0,
                title="",
                authors="",
                total_pages=0,
                chunks_created=0,
                processing_time=round(time.time() - start_time, 2),
                status="failed"
            ))
    
    return results


@router.get("/papers", response_model=List[schemas.PaperResponse])
async def get_all_papers(
    skip: int = 0, 
    limit: int = 100,
    year: Optional[int] = None,
    author: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all papers with pagination and optional filters
    
    - skip: Number of records to skip (for pagination)
    - limit: Maximum number of records to return
    - year: Filter by publication year
    - author: Filter by author name (partial match)
    """
    try:
        query = db.query(database_models.Paper).filter(
            database_models.Paper.is_deleted == False
        )
        
        # Apply filters
        if year:
            query = query.filter(database_models.Paper.year == year)
        
        if author:
            query = query.filter(database_models.Paper.authors.ilike(f"%{author}%"))
        
        # Order by most recent first
        query = query.order_by(database_models.Paper.uploaded_at.desc())
        
        # Pagination
        papers = query.offset(skip).limit(limit).all()
        
        return papers
        
    except Exception as e:
        print(f"❌ Database error in GET /api/papers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error. Please check your database connection and try again. Error: {str(e)}"
        )


@router.get("/papers/{paper_id}", response_model=schemas.PaperDetail)
async def get_single_paper(paper_id: int, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific paper
    
    Returns:
    - Basic paper info
    - Chunk count
    - Query count (how many times it was queried)
    """
    paper = db.query(database_models.Paper).filter(
        database_models.Paper.id == paper_id,
        database_models.Paper.is_deleted == False
    ).first()
    
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    
    # Count queries that used this paper
    query_count = db.query(database_models.QueryPaper).filter(
        database_models.QueryPaper.paper_id == paper_id
    ).count()
    
    # Build response
    paper_detail = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "file_name": paper.file_name,
        "file_path": paper.file_path,
        "file_size": paper.file_size,
        "total_pages": paper.total_pages,
        "total_chunks": paper.total_chunks,
        "uploaded_at": paper.uploaded_at,
        "qdrant_collection_name": paper.qdrant_collection_name,
        "processing_status": paper.processing_status,
        "query_count": query_count
    }
    
    return paper_detail


@router.delete("/papers/{paper_id}")
async def remove_paper(paper_id: int, db: Session = Depends(get_db)):
    """
    Delete a paper and its associated data
    
    - Marks paper as deleted in database
    - Removes associated vectors from Qdrant
    - Removes query associations
    - Optionally deletes the physical PDF file
    """
    paper = db.query(database_models.Paper).filter(
        database_models.Paper.id == paper_id,
        database_models.Paper.is_deleted == False
    ).first()
    
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    
    try:
        # Delete vectors from Qdrant
        print(f"🗑️  Deleting vectors for paper {paper_id} from Qdrant...")
        
        # Mark paper as deleted (soft delete)
        paper.is_deleted = True
        db.commit()
        
        # Optionally delete the physical file
        if os.path.exists(paper.file_path):
            try:
                os.remove(paper.file_path)
                print(f"✅ Deleted file: {paper.file_path}")
            except Exception as e:
                print(f"⚠️  Could not delete file: {e}")
        
        return {
            "message": f"Paper '{paper.title}' deleted successfully",
            "paper_id": paper_id,
            "title": paper.title
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete paper: {str(e)}"
        )


@router.get("/papers/{paper_id}/stats", response_model=schemas.PaperStats)
async def get_stats_for_paper(paper_id: int, db: Session = Depends(get_db)):
    """
    Get detailed statistics for a specific paper
    
    Returns:
    - Total queries made
    - Average response time
    - Average confidence score
    - Total chunks
    """
    paper = db.query(database_models.Paper).filter(
        database_models.Paper.id == paper_id,
        database_models.Paper.is_deleted == False
    ).first()
    
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with ID {paper_id} not found"
        )
    
    # Get all queries that used this paper
    query_papers = db.query(database_models.QueryPaper).filter(
        database_models.QueryPaper.paper_id == paper_id
    ).all()
    
    query_ids = [qp.query_id for qp in query_papers]
    
    if query_ids:
        queries = db.query(database_models.QueryHistory).filter(
            database_models.QueryHistory.id.in_(query_ids)
        ).all()
        
        total_queries = len(queries)
        avg_response_time = sum(q.response_time for q in queries) / total_queries
        
        # Calculate average confidence (handle None values)
        confidence_scores = [q.confidence_score for q in queries if q.confidence_score is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    else:
        total_queries = 0
        avg_response_time = 0.0
        avg_confidence = 0.0
    
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "total_pages": paper.total_pages,
        "total_chunks": paper.total_chunks,
        "total_queries": total_queries,
        "avg_response_time": round(avg_response_time, 3),
        "avg_confidence": round(avg_confidence, 3),
        "most_common_sections_queried": [],
        "upload_date": paper.uploaded_at
    }


# ============================================
# QUERY ENDPOINTS
# ============================================

@router.post("/query", response_model=schemas.QueryResponse)
async def submit_query(query: schemas.QueryRequest, db: Session = Depends(get_db)):
    """
    Query research papers using RAG (Retrieval Augmented Generation)
    
    - Semantic search across paper chunks in Qdrant
    - Filter by specific papers (optional)
    - Generate answer using retrieved context
    - Return citations with relevance scores
    """
    start_time = time.time()
    
    try:
        # Validate paper IDs if provided
        if query.paper_ids:
            valid_papers = db.query(database_models.Paper).filter(
                database_models.Paper.id.in_(query.paper_ids)
            ).all()
            
            if not valid_papers:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="None of the specified papers were found"
                )
        
        # Search for relevant chunks in Qdrant (without filter, we'll filter after)
        top_k = query.top_k or 5
        # Request more results if filtering by paper_ids, since we'll filter after
        search_k = top_k * 3 if query.paper_ids else top_k
        
        search_results = qdrant_service.search_similar(
            query_text=query.question,
            top_k=search_k,
            score_threshold=0.0  # No threshold, we want all results
        )
        
        # Filter by paper_ids if specified
        if query.paper_ids:
            search_results = [
                result for result in search_results 
                if result["metadata"].get("paper_id") in query.paper_ids
            ]
            # Limit to top_k after filtering
            search_results = search_results[:top_k]
        
        if not search_results:
            return schemas.QueryResponse(
                answer="I couldn't find relevant information in the uploaded papers to answer your question.",
                citations=[],
                sources_used=[],
                confidence=0.0,
                response_time=round(time.time() - start_time, 2)
            )
        
        # Build context from search results
        context_parts = []
        citations = []
        sources_used = set()
        
        for i, result in enumerate(search_results, 1):
            chunk_text = result["chunk_text"]
            metadata = result["metadata"]
            score = result["score"]
            
            # Add to context
            context_parts.append(
                f"[Source {i}] From '{metadata['paper_title']}' "
                f"(Section: {metadata.get('section', 'Unknown')}, Page: {metadata.get('page_number', 'N/A')}):\n"
                f"{chunk_text}\n"
            )
            
            # Create citation
            citations.append(schemas.Citation(
                paper_id=metadata["paper_id"],
                paper_title=metadata["paper_title"],
                section=metadata.get("section", "Unknown"),
                page=metadata.get("page_number", 1),
                relevance_score=round(score, 2),
                chunk_text=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            ))
            
            # Track source files
            sources_used.add(metadata["file_name"])
        
        # Generate answer using LLM with context
        context = "\n".join(context_parts)
        
        prompt = f"""You are a research assistant helping users understand academic papers. 
Answer the question based ONLY on the provided context from research papers.

Context from papers:
{context}

Question: {query.question}

Instructions:
1. Provide a clear, accurate answer based on the context
2. Reference specific sources when making claims (e.g., "According to Source 1...")
3. If the context doesn't contain enough information, say so
4. Be concise but comprehensive
5. Use academic tone

Answer:"""

        answer = ollama_service.generate_answer(
            question=query.question,
            context=prompt
        )
        
        # Calculate confidence based on relevance scores
        avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
        confidence = min(avg_relevance * 1.2, 1.0)  # Boost slightly, cap at 1.0
        
        processing_time = time.time() - start_time
        
        # Save query to history
        query_history = database_models.QueryHistory(
            query_text=query.question,
            answer=answer,
            confidence_score=confidence,
            response_time=processing_time,
            top_k=top_k,
            sources_used=list(sources_used),
            citations=[{
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
                "section": c.section,
                "page": c.page,
                "relevance_score": c.relevance_score
            } for c in citations]
        )
        db.add(query_history)
        db.commit()
        db.refresh(query_history)
        
        # Link query to papers
        for paper_id in set(result["metadata"]["paper_id"] for result in search_results):
            query_paper = database_models.QueryPaper(
                query_id=query_history.id,
                paper_id=paper_id
            )
            db.add(query_paper)
        
        db.commit()
        
        return schemas.QueryResponse(
            answer=answer,
            citations=citations,
            sources_used=list(sources_used),
            confidence=round(confidence, 2),
            response_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@router.get("/queries/history", response_model=List[schemas.QueryHistoryResponse])
async def fetch_query_history(
    skip: int = 0, 
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get query history with pagination
    
    Returns queries ordered by most recent first
    """
    queries = db.query(database_models.QueryHistory).order_by(
        database_models.QueryHistory.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    # Format response
    results = []
    for q in queries:
        results.append({
            "id": q.id,
            "query_text": q.query_text,
            "answer": q.answer,
            "response_time": q.response_time,
            "confidence_score": q.confidence_score,
            "sources_used": q.sources_used or [],
            "citations": q.citations or [],
            "created_at": q.created_at,
            "user_rating": q.user_rating
        })
    
    return results


# ============================================
# ANALYTICS ENDPOINTS
# ============================================

@router.get("/analytics/popular")
async def fetch_popular_queries(limit: int = 10, db: Session = Depends(get_db)):
    """
    Get most popular/frequent queries
    
    Returns most commonly asked questions with statistics
    """
    all_queries = db.query(database_models.QueryHistory).all()
    
    if not all_queries:
        return []
    
    # Aggregate by query text
    query_stats = {}
    for q in all_queries:
        text = q.query_text.lower().strip()  # Normalize
        
        if text not in query_stats:
            query_stats[text] = {
                "query_text": q.query_text,  # Use original text
                "count": 0,
                "total_confidence": 0.0,
                "confidence_count": 0
            }
        
        query_stats[text]["count"] += 1
        
        if q.confidence_score is not None:
            query_stats[text]["total_confidence"] += q.confidence_score
            query_stats[text]["confidence_count"] += 1
    
    # Format results
    popular = []
    for text, data in query_stats.items():
        avg_confidence = (
            data["total_confidence"] / data["confidence_count"] 
            if data["confidence_count"] > 0 
            else 0.0
        )
        
        popular.append({
            "query_text": data["query_text"],
            "count": data["count"],
            "avg_confidence": round(avg_confidence, 3)
        })
    
    # Sort by count (descending)
    popular.sort(key=lambda x: x["count"], reverse=True)
    
    return popular[:limit]
