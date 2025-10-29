import os
import shutil
from qdrant_client import QdrantClient
from src.database import SessionLocal
from src.models import database_models


def cleanup_all_data():
    """
    Clean up all data from PostgreSQL and Qdrant before server starts.
    This ensures assessors see fresh API functionality without pre-existing data.
    """
    print("\n" + "="*60)
    print("🧹 CLEANUP: Removing all existing data for fresh assessment...")
    print("="*60)
    
    # 1. Clean PostgreSQL database
    try:
        db = SessionLocal()
        
        # Delete all records in correct order (respect foreign keys)
        deleted_query_papers = db.query(database_models.QueryPaper).delete()
        deleted_queries = db.query(database_models.QueryHistory).delete()
        deleted_papers = db.query(database_models.Paper).delete()
        
        db.commit()
        db.close()
        
        print(f"✅ PostgreSQL cleanup complete:")
        print(f"   - Deleted {deleted_query_papers} query-paper associations")
        print(f"   - Deleted {deleted_queries} query history records")
        print(f"   - Deleted {deleted_papers} paper records")
    except Exception as e:
        print(f"⚠️  PostgreSQL cleanup error: {e}")
        try:
            db.rollback()
            db.close()
        except:
            pass
    
    # 2. Clean Qdrant vector database
    try:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "research_papers")
        
        client = QdrantClient(url=qdrant_url)
        
        # Delete collection if it exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if collection_exists:
            client.delete_collection(collection_name=collection_name)
            print(f"✅ Qdrant cleanup complete:")
            print(f"   - Deleted collection '{collection_name}'")
        else:
            print(f"ℹ️  Qdrant collection '{collection_name}' does not exist (already clean)")
            
    except Exception as e:
        print(f"⚠️  Qdrant cleanup error: {e}")
    
    # 3. Clean uploaded files
    try:
        upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        if os.path.exists(upload_dir):
            # Remove all files in uploads directory
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"⚠️  Could not delete {filename}: {e}")
            print(f"✅ Uploads directory cleaned")
        else:
            os.makedirs(upload_dir, exist_ok=True)
            print(f"✅ Uploads directory created")
    except Exception as e:
        print(f"⚠️  Upload directory cleanup error: {e}")
    
    print("="*60)
    print("✨ Cleanup complete! Ready for fresh assessment.")
    print("="*60 + "\n")
