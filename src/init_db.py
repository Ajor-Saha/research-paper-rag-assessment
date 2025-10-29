"""
Database initialization script
Run this to create all tables in your PostgreSQL database
"""

from src.database import engine, Base
from src.models.database_models import Paper, QueryHistory, QueryPaper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """
    Create all database tables
    """
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
        
        # Print created tables
        logger.info("\nCreated tables:")
        for table_name in Base.metadata.tables.keys():
            logger.info(f"  - {table_name}")
            
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise


def drop_all_tables():
    """
    Drop all tables (use with caution!)
    """
    try:
        logger.warning("⚠️  Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ All tables dropped successfully!")
    except Exception as e:
        logger.error(f"❌ Error dropping tables: {e}")
        raise


def reset_database():
    """
    Drop and recreate all tables (use with caution!)
    """
    logger.warning("⚠️  Resetting database...")
    drop_all_tables()
    init_database()
    logger.info("✅ Database reset complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            init_database()
        elif command == "drop":
            confirm = input("Are you sure you want to drop all tables? (yes/no): ")
            if confirm.lower() == "yes":
                drop_all_tables()
            else:
                logger.info("Operation cancelled")
        elif command == "reset":
            confirm = input("Are you sure you want to reset the database? This will delete all data! (yes/no): ")
            if confirm.lower() == "yes":
                reset_database()
            else:
                logger.info("Operation cancelled")
        else:
            print("Unknown command. Use: init, drop, or reset")
    else:
        # Default: just initialize
        init_database()
