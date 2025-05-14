"""
Query processing service.

This module handles document querying, LLM interactions, and response generation.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from controllers.ask import get_llm_response, get_general_llm_response, get_demo_response
from controllers.database import is_user_limit_over, is_trial_limit_over
from utils.guardrails import input_guardrail_pipeline

logger = logging.getLogger(__name__)

class QueryService:
    """Service for handling document queries and LLM interactions."""
    
    def process_trial_query(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process a query from a trial user.
        
        Args:
            data: Query data including fingerprint, message, and language settings
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        
        # Check free trial limits
        try:
            fingerprint = data.get('fingerprint')
            if not fingerprint:
                return {'message': 'Fingerprint is missing'}, 400
                
            if is_trial_limit_over(fingerprint):
                return {'message': 'Free Trial limit is exhausted'}, 200
        except Exception as e:
            logger.exception(f'Error validating fingerprint: {e}')
            return {'message': 'Error validating fingerprint'}, 400
        
        # Extract query parameters
        try:
            user_query = data.get('message')
            input_language = int(data.get('inputLanguage'))
            output_language = int(data.get('outputLanguage'))
            context = data.get('context')
            hascsvxl = data.get('hasCsvOrXlsx')
            mode = data.get('mode')
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Generate response using LLM
        try:
            logger.info(f'User query: {user_query}')
            session_name = fingerprint
            
            if context:
                llm_response = get_llm_response(
                    user_query, input_language, output_language, 
                    session_name, hascsvxl=hascsvxl, mode=mode
                )
            else:
                llm_response = get_general_llm_response(
                    user_query, input_language, output_language, session_name
                )
        except Exception as e:
            logger.exception(f'Error generating LLM response: {e}')
            return {'message': 'Error generating response from LLM'}, 500

        logger.info(f'Trial query response: {llm_response}')
        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return llm_response, 200
    
    def process_authenticated_query(self, data: Dict[str, Any], user_email: str, session_name: str) -> Tuple[Dict[str, Any], int]:
        """
        Process a query from an authenticated user.
        
        Args:
            data: Query data including message and language settings
            session_name: User session identifier
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        
        # Check if the user has exceeded query limits
        if is_user_limit_over(user_email):
            return {"answer": "To ask further questions, please upgrade your account."}, 200
        
        # Extract query parameters
        try:
            user_query = data.get('message')
            input_language = int(data.get('inputLanguage'))
            output_language = int(data.get('outputLanguage'))
            context = data.get('context')
            hascsvxl = data.get('hasCsvOrXlsx')
            mode = data.get('mode')
            filenames = data.get('filenames', [])
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Generate response using LLM
        try:
            logger.info(f'User query: {user_query}')

            # Input guardrail processing
            input_guardrail = input_guardrail_pipeline()
            guardrail_response = input_guardrail.process_input(user_query)
            if guardrail_response.get("status") == "blocked":
                # If the input is blocked, return the guardrail response
                return guardrail_response, 400
            else:
                # If the input is allowed, proceed with the LLM response
                user_query = guardrail_response.get("sanitized_input")
                logger.info(f'Sanitized user query: {user_query}')
            
            if context:
                llm_response = get_llm_response(
                    user_query, input_language, output_language, 
                    session_name, hascsvxl=hascsvxl, mode=mode, filenames=filenames
                )
            else:
                llm_response = get_general_llm_response(
                    user_query, input_language, output_language, session_name
                )
        except Exception as e:
            logger.exception(f'Error generating LLM response: {e}')
            return {'message': 'Error generating response from LLM'}, 500

        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return llm_response, 200
    
    def process_demo_query(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process a demo query for public transport information.
        
        Args:
            data: Query data including message
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        
        # Extract query parameters
        try:
            user_query = data.get('message')
            if not user_query:
                return {'message': 'Query is missing'}, 400
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Generate response using LLM
        try:
            logger.info(f'Demo query: {user_query}')
            session_name = "user20250317t114507"
            llm_response = get_demo_response(user_query, session_name)
        except Exception as e:
            logger.exception(f'Error generating demo response: {e}')
            return {'message': 'Error generating response from LLM'}, 500

        logger.info(f'Demo response: {llm_response}')
        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return {"answer": llm_response}, 200

# Create a singleton instance
query_service = QueryService()

def get_query_service() -> QueryService:
    """
    Get the query service singleton instance.
    
    Returns:
        QueryService: The query service instance
    """
    return query_service