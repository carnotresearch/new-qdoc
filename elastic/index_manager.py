from .client import ElasticClient
import logging

class ElasticIndexManager:
    def __init__(self):
        self.client = ElasticClient().client
        self.default_mapping = {
            "mappings": {
                "properties": {
                    "vector": {"type": "dense_vector", "dims": 3072},
                    "content": {"type": "text"},
                    "keyword_content": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "filename": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "header": {"type": "text"}
                        }
                    }
                }
            }
        }

    def create_index(self, index_name, mapping=None):
        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(
                    index=index_name,
                    body=mapping or self.default_mapping
                )
                logging.info(f"Index '{index_name}' created")
                return True
            return False
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            raise

    def delete_index(self, index_name):
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logging.info(f"Index '{index_name}' deleted")
                return True
            return False
        except Exception as e:
            logging.error(f"Error deleting index: {e}")
            raise

    def refresh_index(self, index_name):
        try:
            self.client.indices.refresh(index=index_name)
            return True
        except Exception as e:
            logging.error(f"Error refreshing index: {e}")
            raise
    
    def index_exists(self, index_name):
        """Check if index exists using the ElasticClient's verification"""
        try:
            return self.client.indices.exists(index=index_name)
        except Exception as e:
            logging.error(f"Index existence check failed: {e}")
            return False