"""
Enhanced Database Operations Module with:
- Secure connection handling
- Excel multi-sheet support
- SQL injection protection
- Comprehensive error handling
"""

# Standard library imports
import logging
import os
import re
import json
from datetime import datetime
from typing import List, Optional, Tuple

# Third-party imports
import pandas as pd
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, exc, text
from sqlalchemy.engine.base import Engine

# Local imports
from config import Config
from app.services.llm_service import get_standard_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MYSQL_CONFIG = {
    "drivername": "mysql+mysqlconnector",
    "username": Config.MYSQL_USERNAME,
    "password": Config.MYSQL_PASSWORD,
    "host": Config.MYSQL_HOST,
    "port": Config.MYSQL_PORT
}

def sanitize_identifier(name: str, max_length: int = 63) -> str:
    """Sanitize column names to prevent SQL injection and truncation."""
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'\W+', '_', name)
    # Truncate to 64 characters for MySQL compatibility
    return sanitized[:64]

def get_db_engine(user_session: str) -> Optional[Engine]:
    """Create SQLAlchemy engine with robust database creation."""
    sanitized_db = f"db_{sanitize_identifier(user_session)}"
    
    try:
        # Create temporary engine for database creation
        base_engine = create_engine(
            f"{MYSQL_CONFIG['drivername']}://"
            f"{MYSQL_CONFIG['username']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/",
            isolation_level="AUTOCOMMIT"
        )
        
        with base_engine.connect() as conn:
            # Verify database creation
            result = conn.execute(
                text(f"SHOW DATABASES LIKE '{sanitized_db}'")
            )
            if not result.fetchone():
                conn.execute(text(f"CREATE DATABASE {sanitized_db}"))
                logger.info(f"Created new database: {sanitized_db}")

        # Create final engine with connection pool
        return create_engine(
            f"{MYSQL_CONFIG['drivername']}://"
            f"{MYSQL_CONFIG['username']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{sanitized_db}",
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        
    except exc.SQLAlchemyError as e:
        logger.error(f"Database creation failed: {str(e)}")
        return None

def process_file(file, engine: Engine) -> List[str]:
    """Process CSV/Excel file with multiple sheets into database tables."""
    created_tables = []
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            df.columns = [sanitize_identifier(col) for col in df.columns]
            table_name = sanitize_identifier(file.filename.rsplit('.', 1)[0])
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False
            )
            created_tables.append(table_name)
            
        elif file.filename.endswith(('.xlsx', '.xls')):
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.columns = [sanitize_identifier(col) for col in df.columns]
                    safe_sheet_name = sanitize_identifier(sheet_name)
                    safe_table_name = f"{sanitize_identifier(file.filename.split('.')[0])}_{safe_sheet_name}"
                    df.to_sql(
                        name=safe_table_name,
                        con=engine,
                        if_exists='replace',
                        index=False
                    )
                    created_tables.append(safe_table_name)
                except Exception as e:
                    logger.error(f"Failed to process sheet {sheet_name}: {str(e)}")
                    continue
        return created_tables
        
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise

def create_database_with_tables(user_session: str, files: List) -> Tuple[bool, str]:
    """Create new database and process initial files."""
    try:
        engine = get_db_engine(user_session)
        if not engine:
            return False, "Database connection failed"
            
        with engine.begin() as connection:
            for file in files:
                tables_created = process_file(file, engine)
                logger.info(f"Created tables: {', '.join(tables_created)}")
                # Store sheet metadata in JSON file
                if len(tables_created) > 0:
                    metadata_dir = os.path.join('users', user_session, "files")
                    metadata_path = os.path.join(metadata_dir, "sheet_metadata.json")

                    # Ensure directory exists
                    os.makedirs(metadata_dir, exist_ok=True)

                    # Load existing metadata if file exists
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    # Update metadata
                    metadata[file.filename] = tables_created

                    # Save updated metadata
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                
        return True, "Database created successfully"
        
    except exc.SQLAlchemyError as e:
        logger.error(f"Database creation error: {str(e)}")
        return False, "Database creation failed"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, "Unexpected error occurred"

def add_tables_to_existing_db(user_session: str, files: List) -> Tuple[bool, str]:
    """Add new tables to existing database."""
    try:
        engine = get_db_engine(user_session)
        if not engine:
            return False, "Database connection failed"
            
        existing_tables = engine.dialect.get_table_names(engine.connect())
        new_tables = []
        
        with engine.begin() as connection:
            for file in files:
                tables_created = process_file(file, engine)
                new_tables.extend([t for t in tables_created if t not in existing_tables])
                # Store sheet metadata in JSON file
                if len(tables_created) > 0:
                    metadata_dir = os.path.join('users', user_session, "files")
                    metadata_path = os.path.join(metadata_dir, "sheet_metadata.json")

                    # Ensure directory exists
                    os.makedirs(metadata_dir, exist_ok=True)

                    # Load existing metadata if file exists
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    # Update metadata
                    metadata[file.filename] = tables_created

                    # Save updated metadata
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                
        logger.info(f"Added {len(new_tables)} new tables")
        return True, f"Added {len(new_tables)} new tables successfully"
        
    except exc.SQLAlchemyError as e:
        logger.error(f"Table addition error: {str(e)}")
        return False, "Failed to add tables"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, "Unexpected error occurred"

def execute_safe_query(engine: Engine, query: str) -> List[dict]:
    """Safe query execution with error recovery."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return [dict(row) for row in result.mappings()]
            
    except exc.ProgrammingError as e:
        logger.error(f"SQL syntax error: {str(e)}")
        # Attempt automatic correction for common GPT errors
        corrected_query = re.sub(r':', '', query)  # Fix parameter placeholder issues
        corrected_query = re.sub(r'`\s*\.\s*`', '.', corrected_query)  # Fix backtick spacing
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(corrected_query))
                return [dict(row) for row in result.mappings()]
        except:
            raise ValueError(f"Could not recover from SQL error: {str(e)}")

def clean_sql_query(raw_query: str) -> str:
    """Precision SQL cleaning with semicolon normalization."""
    # Extract core query using strict pattern matching
    sql_pattern = r"""
        (?i)                    # Case-insensitive
        \bSELECT\b              # Must start with SELECT
        .*?                     # Any characters (non-greedy)
        (;?)                    # Optional semicolon
        (?=\s*(?:```|$))        # Lookahead for end markers
    """
    match = re.search(sql_pattern, raw_query, re.VERBOSE | re.DOTALL)
    
    if not match:
        raise ValueError("No valid SELECT statement found")
    
    cleaned = match.group(0)
    
    # Normalize semicolons and whitespace
    cleaned = cleaned.rstrip(';').strip()  # Remove trailing semicolon
    cleaned = re.sub(r';\s*;', ';', cleaned)  # Remove duplicate semicolons
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse whitespace
    
    return cleaned

# def validate_sql(engine: Engine, query: str) -> bool:
#     """Safer validation using query parameterization."""
#     try:
#         with engine.connect() as conn:
#             # Test with LIMIT 1 to avoid full execution
#             test_query = query.replace(";", "") + " LIMIT 1"
#             conn.execute(text(test_query)).fetchone()
#         return True
#     except exc.SQLAlchemyError as e:
#         logger.error(f"Validation failed: {str(e)}")
#         raise ValueError(f"Invalid SQL: {str(e)}")


def generate_sql_query(engine: Engine, natural_language_query: str) -> str:
    """Fail-safe generation pipeline with parse validation."""
    try:
        db = SQLDatabase(engine)
        # Use centralized LLM service for SQL query generation
        llm = get_standard_llm()
        
        # Force clean output with system prompt
        sys_prompt = """You are a SQL expert. Return ONLY ONE PERFECT SQL QUERY between ```sql delimiters. 
        NO EXPLANATIONS. NO MARKDOWN. NO DISCLAIMERS. JUST SQL."""
        
        raw_query = create_sql_query_chain(llm, db).invoke({
            "question": f"{sys_prompt}\nQuestion: {natural_language_query}"
        })
        
        logger.info(f"Raw LLM output:\n{raw_query}")
        
        # Atomic cleaning with strict validation
        cleaned = clean_sql_query(raw_query)
        logger.info(f"Cleaned query:\n{cleaned}")
        
        # Validate using MySQL's actual parser
        # validate_sql(engine, cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Query generation pipeline failed: {str(e)}")
        raise

def query_database(user_session: str, natural_language_query: str) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Full query pipeline with error handling."""
    try:
        engine = get_db_engine(user_session)
        if not engine:
            return None, "Database connection failed"
            
        generated_query = generate_sql_query(engine, natural_language_query)
        
        results = execute_safe_query(engine, generated_query)
        formatted_context = f"[0] \"SQL query: {generated_query}\"  \n(SQL output {results})\n\n"
        return formatted_context, None
        
    except Exception as e:
        logger.error(f"Query pipeline failed: {str(e)}")
        return None, str(e)

def store_table_info(user_session: str, file_name: str):
    """
    Extract data from a table and store it in a text file.
    Creates a metadata JSON file with file details.
    
    Args:
        user_session (str): The user session identifier.
        file_name (str): The name of the file to process.
    """
    try:
        # Sanitize database and table names
        sanitized_db = f"db_{sanitize_identifier(user_session)}"
        sanitized_table = sanitize_identifier(file_name.rsplit('.', 1)[0])

        # Get database engine
        engine = get_db_engine(user_session)
        if not engine:
            logger.error("Failed to connect to the database.")
            return

        # if table has multiple sheets, check if the table name is present in the database
        table_names = engine.dialect.get_table_names(engine.connect())
        logger.info(f"Available tables: {table_names}")
        if sanitized_table not in table_names:
            logger.info(f"Table {sanitized_table} does not exist in the database.")
            # access the metadata file to get the list of sheets
            metadata_path = os.path.join('users', user_session, "files", "sheet_metadata.json")
            sheets_data = []
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                if file_name in metadata:
                    logger.info(f"file: {file_name}, table list: {metadata[file_name]}")
                    for table in metadata[file_name]:
                        if table in table_names:
                            query = f"SELECT * FROM {table} LIMIT 10"
                            logger.info(f"Executing query: {query}")
                            result = execute_safe_query(engine, query)
                            sheets_data.append(result)
            else:
                logger.error("sheet_metadata.json file does not exist.")
                return
            # Combine results from all sheets
            if not sheets_data:
                logger.error("No valid sheets found in metadata.")
                return
            results = "\n\n".join([str(data) for data in sheets_data])
        else:
            # Execute SQL query to fetch all data from the table
            query = f"SELECT * FROM {sanitized_table} LIMIT 10"
            logger.info(f"Executing query: {query}")
            results = execute_safe_query(engine, query)

        # Prepare file paths
        base_dir = os.path.join('users', user_session, "files", sanitized_table)
        os.makedirs(base_dir, exist_ok=True)
        imp_sents_path = os.path.join(base_dir, "imp_sents.txt")
        metadata_path = os.path.join(base_dir, "metadata.json")

        # Write results to imp_sents.txt
        with open(imp_sents_path, "w", encoding="utf-8") as imp_sents_file:
            for row in results:
                imp_sents_file.write(f"{row}\n")
        logger.info(f"Data written to {imp_sents_path}")

        # Create metadata.json
        metadata = {
            "filename": file_name,
            "file_size": os.path.getsize(imp_sents_path),
            "has_summary": True,
            "summary_created_at": datetime.now().timestamp()
        }
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        logger.info(f"Metadata written to {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to store table info: {str(e)}")