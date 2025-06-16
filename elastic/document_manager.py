from langchain.schema import Document
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_openai.embeddings import OpenAIEmbeddings
from .index_manager import ElasticIndexManager
from .client import ElasticClient
import logging
import unicodedata
from config import Config

class ElasticDocumentManager:
    def __init__(self, index_name):
        self.index_name = index_name
        self.client = ElasticClient().client
        self.index_manager = ElasticIndexManager()
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            api_key=Config.OPENAI_API_KEY,
        )

    def store_documents(self, documents):
        try:
            # Pre-process documents for better Unicode handling
            processed_documents = self._preprocess_documents(documents)
            
            # Create index if not exists
            self.index_manager.create_index(self.index_name)
            
            # Store documents with embeddings
            vector_store = ElasticsearchStore.from_documents(
                processed_documents,
                es_cloud_id=Config.ES_CLOUD_ID,
                index_name=self.index_name,
                es_api_key=Config.ES_API_KEY,
                embedding=self.embeddings,
                vector_query_field="vector"
            )
            
            # Refresh index for immediate visibility
            self.index_manager.refresh_index(self.index_name)
            logging.info(f"Stored {len(processed_documents)} documents in {self.index_name}")
            return vector_store
        except Exception as e:
            logging.error(f"Error storing documents: {e}")
            raise

    def _preprocess_documents(self, documents):
        """Preprocess documents for better Unicode/Hindi handling"""
        processed = []
        
        for doc in documents:
            try:
                # Ensure proper Unicode normalization
                content = doc.page_content
                if content:
                    # Normalize Unicode
                    content = unicodedata.normalize("NFC", content)
                    
                    # Ensure UTF-8 encoding
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='ignore')
                    
                    # Create new document with processed content
                    processed_doc = Document(
                        page_content=content,
                        metadata=doc.metadata
                    )
                    processed.append(processed_doc)
                    
            except Exception as e:
                logging.error(f"Error preprocessing document: {e}")
                # Include original document if preprocessing fails
                processed.append(doc)
        
        return processed

    def update_documents(self, documents):
        # Implementation for document updates
        pass

    def delete_documents(self, document_ids):
        # Implementation for document deletion
        pass