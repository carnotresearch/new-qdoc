from langchain.retrievers import EnsembleRetriever
from langchain_elasticsearch.retrievers import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from .client import ElasticClient
from .index_manager import ElasticIndexManager
import logging
from config import Config

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

    def create_ensemble_retriever(self, weights=(0.4, 0.6)):
        try:
            if not self.index_manager.index_exists(self.index_name):
                return None

            # Keyword retriever
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
            rewrite_prompt = f"""Rewrite this query for better document retrieval:
                1. Fix spelling/grammar
                2. Expand acronyms
                3. Add implicit context
                4. Include synonyms
                5. Maintain original intent
                
                Original: {query}

                Do not include any other text or explanations.
                """
            
            llm = ChatOpenAI(model="gpt-4o", api_key=self.openai_api_key)
            response = llm.invoke(rewrite_prompt).content
            logging.info(f"Query enrichment response: {response}")
            if not response:
                logging.warning("Query enrichment returned empty response.")
                return None
            return response.strip()
        except Exception as e:
            logging.error(f"Query enrichment error: {e}")
            return None

    def _create_filtered_query(self, query):
        # Query enrichment
        enriched_query = self.query_enrichment(query)
        
        should_clauses = [{"match": {"text": query}}]
        if enriched_query:
            should_clauses.append({"match": {"text": enriched_query}})
        return {
            "query": {
                "bool": {
                    "should": should_clauses
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