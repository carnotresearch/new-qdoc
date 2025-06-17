from langchain.retrievers import EnsembleRetriever
from langchain_elasticsearch.retrievers import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai.embeddings import OpenAIEmbeddings
from .client import ElasticClient
from .index_manager import ElasticIndexManager
import logging
import unicodedata
from config import Config
from app.services.llm_service import get_fast_llm

class ElasticRetriever:
    def __init__(self, index_name):
        self.index_name = index_name
        self.client = ElasticClient().client
        self.index_manager = ElasticIndexManager()
        self.openai_api_key = Config.OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            api_key=self.openai_api_key,
        )
        # Use fast LLM for query enrichment
        self.llm = get_fast_llm()

    def create_ensemble_retriever(self, weights=(0.4, 0.6)):
        try:
            if not self.index_manager.index_exists(self.index_name):
                return None

            # Keyword retriever with Unicode support
            key_retriever = ElasticsearchRetriever(
                es_client=self.client,
                index_name=self.index_name,
                body_func=self._create_filtered_query,
                content_field="text"
            )

            # Vector retriever
            vector_store = ElasticsearchStore(
                es_cloud_id=Config.ES_CLOUD_ID,
                es_api_key=Config.ES_API_KEY,
                index_name=self.index_name,
                embedding=self.embeddings,
            )
            vector_retriever = vector_store.as_retriever()

            return EnsembleRetriever(
                retrievers=[key_retriever, vector_retriever],
                weights=list(weights)
            )
        except Exception as e:
            logging.error(f"Error creating retriever: {e}")
            raise
    
    def query_enrichment(self, query):
        try:
            # Normalize query for better Unicode handling
            normalized_query = unicodedata.normalize("NFC", query)
            
            rewrite_prompt = f"""Rewrite this query for better document retrieval:
                1. Fix spelling/grammar
                2. Expand acronyms
                3. Add implicit context
                4. Include synonyms
                5. Maintain original intent
                6. If query is in Hindi, preserve the Hindi script
                
                Original: {normalized_query}

                Do not include any other text or explanations.
                """
            
            response = self.llm.invoke(rewrite_prompt).content
            logging.info(f"Query enrichment response: {response}")
            if not response:
                logging.warning("Query enrichment returned empty response.")
                return None
            return response.strip()
        except Exception as e:
            logging.error(f"Query enrichment error: {e}")
            return None

    def _create_filtered_query(self, query):
        """Create search query with better Unicode support"""
        # Normalize the query
        normalized_query = unicodedata.normalize("NFC", query)
        
        # Query enrichment
        enriched_query = self.query_enrichment(normalized_query)
        
        should_clauses = [
            {"match": {"text": normalized_query}},
            {"match": {"text.hindi": normalized_query}}  # Use Hindi analyzer field
        ]
        
        if enriched_query and enriched_query != normalized_query:
            should_clauses.extend([
                {"match": {"text": enriched_query}},
                {"match": {"text.hindi": enriched_query}}
            ])
        
        return {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            },
            "_source": ["text", "metadata.source", "metadata.page", "metadata.filename"]
        }

    def search(self, query, k=4):
        try:
            retriever = self.create_ensemble_retriever()
            if not retriever:
                return None
            return retriever.invoke(query, k=k)
        except Exception as e:
            logging.error(f"Search error: {e}")
            raise