from langchain.schema import Document
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_openai.embeddings import OpenAIEmbeddings
from .index_manager import ElasticIndexManager
from .client import ElasticClient
import logging
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
            # Create index if not exists
            self.index_manager.create_index(self.index_name)
            
            # Store documents with embeddings
            vector_store = ElasticsearchStore.from_documents(
                documents,
                es_cloud_id=Config.ES_CLOUD_ID,
                index_name=self.index_name,
                es_api_key=Config.ES_API_KEY,
                embedding=self.embeddings,
                vector_query_field="vector"
            )
            
            # Refresh index for immediate visibility
            self.index_manager.refresh_index(self.index_name)
            logging.info(f"Stored {len(documents)} documents in {self.index_name}")
            return vector_store
        except Exception as e:
            logging.error(f"Error storing documents: {e}")
            raise

    def update_documents(self, documents):
        # Implementation for document updates
        pass

    def delete_documents(self, document_ids):
        # Implementation for document deletion
        pass