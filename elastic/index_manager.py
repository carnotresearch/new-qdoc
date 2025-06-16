from .client import ElasticClient
import logging

class ElasticIndexManager:
    def __init__(self):
        self.client = ElasticClient().client
        # Enhanced mapping with better Unicode support
        self.default_mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "hindi_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "stop",
                                "hindi_normalization"
                            ]
                        },
                        "multilingual_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding"
                            ]
                        }
                    },
                    "filter": {
                        "hindi_normalization": {
                            "type": "icu_normalizer",
                            "name": "nfc"
                        }
                    }
                },
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector", 
                        "dims": 3072
                    },
                    "text": {
                        "type": "text",
                        "analyzer": "multilingual_analyzer",
                        "fields": {
                            "hindi": {
                                "type": "text",
                                "analyzer": "hindi_analyzer"
                            },
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "multilingual_analyzer"
                    },
                    "keyword_content": {
                        "type": "keyword",
                        "ignore_above": 256
                    },
                    "metadata": {
                        "properties": {
                            "filename": {
                                "type": "keyword"
                            },
                            "source": {
                                "type": "keyword"
                            },
                            "page": {
                                "type": "integer"
                            },
                            "header": {
                                "type": "text",
                                "analyzer": "multilingual_analyzer"
                            }
                        }
                    }
                }
            }
        }

    def create_index(self, index_name, mapping=None):
        try:
            if not self.client.indices.exists(index=index_name):
                # Use enhanced mapping with Unicode support
                mapping_to_use = mapping or self.default_mapping
                
                self.client.indices.create(
                    index=index_name,
                    body=mapping_to_use
                )
                logging.info(f"Index '{index_name}' created with Unicode support")
                return True
            return False
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            # Fallback to simpler mapping if ICU plugin is not available
            if "icu_normalizer" in str(e):
                logging.warning("ICU plugin not available, using fallback mapping")
                fallback_mapping = self._get_fallback_mapping()
                try:
                    self.client.indices.create(
                        index=index_name,
                        body=fallback_mapping
                    )
                    logging.info(f"Index '{index_name}' created with fallback mapping")
                    return True
                except Exception as e2:
                    logging.error(f"Fallback mapping also failed: {e2}")
                    raise
            raise

    def _get_fallback_mapping(self):
        """Fallback mapping without ICU plugin"""
        return {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "multilingual_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "vector": {"type": "dense_vector", "dims": 3072},
                    "text": {
                        "type": "text",
                        "analyzer": "multilingual_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "content": {"type": "text", "analyzer": "multilingual_analyzer"},
                    "keyword_content": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "filename": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "header": {"type": "text", "analyzer": "multilingual_analyzer"}
                        }
                    }
                }
            }
        }

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