"""
Query processing service module (updated).

This module handles document querying, LLM interactions, and response generation.
Updated to use the agent-based approach for more efficient and modular processing.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from app.services.query_agent_service import get_query_agent_service
from controllers.database import is_user_limit_over, is_trial_limit_over
from utils.guardrails import input_guardrail_pipeline

# Configure logging
logger = logging.getLogger(__name__)

class QueryService:
    """Service for handling document queries and LLM interactions."""
    
    def __init__(self):
        """Initialize query service."""
        logger.info("Initializing query service")
        self._guardrail = input_guardrail_pipeline()
        self.query_agent = get_query_agent_service()
    
    def process_trial_query(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process a query from a trial user.
        
        Args:
            data: Query data including fingerprint, message, and language settings
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        logger.info("Processing trial query")
        
        # Check free trial limits
        try:
            fingerprint = data.get('fingerprint')
            if not fingerprint:
                logger.warning("Trial query missing fingerprint")
                return {'message': 'Fingerprint is missing'}, 400
                
            if is_trial_limit_over(fingerprint):
                logger.info(f"Trial limit exhausted for fingerprint {fingerprint}")
                return {'message': 'Free Trial limit is exhausted'}, 200
        except Exception as e:
            logger.exception(f'Error validating fingerprint: {e}')
            return {'message': 'Error validating fingerprint'}, 400
        
        # Extract query parameters
        try:
            user_query = self._extract_query_parameters(data)
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Apply guardrails
        try:
            guardrail_response = self._guardrail.process_input(user_query["message"])
            if guardrail_response.get("status") == "blocked":
                logger.warning(f"Query blocked by guardrails: {user_query['message']}")
                return guardrail_response, 400
            else:
                # If the input is allowed, proceed with sanitized input
                user_query["message"] = guardrail_response.get("sanitized_input")
                logger.info(f'Sanitized user query: {user_query["message"]}')
        except Exception as e:
            logger.warning(f"Guardrail error (proceeding with original query): {e}")
        
        # Process query using agent
        try:
            logger.info(f'User trial query: {user_query["message"]}')
            session_name = fingerprint
            
            # Process with query agent
            response = self.query_agent.process_query(
                user_query["message"],
                session_name,
                user_query["input_language"],
                user_query["output_language"],
                user_query["filenames"],
                user_query["hascsvxl"],
                user_query["mode"],
                is_trial=True
            )
        except Exception as e:
            logger.exception(f'Error processing query with agent: {e}')
            return {'message': 'Error generating response'}, 500

        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return response, 200
    
    def process_authenticated_query(self, data: Dict[str, Any], user_email: str, session_name: str) -> Tuple[Dict[str, Any], int]:
        """
        Process a query from an authenticated user.
        
        Args:
            data: Query data including message and language settings
            user_email: User's email
            session_name: User session identifier
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        logger.info(f"Processing authenticated query for user {user_email}")
        
        # Check if the user has exceeded query limits
        if is_user_limit_over(user_email):
            logger.info(f"User {user_email} has exceeded query limits")
            return {"answer": "To ask further questions, please upgrade your account."}, 200
        
        # Extract query parameters
        try:
            user_query = self._extract_query_parameters(data)
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Apply guardrails
        try:
            guardrail_response = self._guardrail.process_input(user_query["message"])
            if guardrail_response.get("status") == "blocked":
                logger.warning(f"Query blocked by guardrails: {user_query['message']}")
                return guardrail_response, 400
            else:
                # If the input is allowed, proceed with sanitized input
                user_query["message"] = guardrail_response.get("sanitized_input")
                logger.info(f'Sanitized user query: {user_query["message"]}')
        except Exception as e:
            logger.warning(f"Guardrail error (proceeding with original query): {e}")
        
        # Process query using agent
        try:
            # Check if context is available
            has_context = user_query.get("context", False)
            logger.info(f"Query has context: {has_context}")
            
            if not has_context:
                # User hasn't selected any files/sessions, use the no-context handler
                logger.info("Processing as a no-context query (general chat)")
                response = self.query_agent.process_no_context_query(
                    user_query["message"],
                    user_email,
                    user_query["input_language"],
                    user_query["output_language"]
                )
            else:
                # User has selected files/session, use normal processing
                logger.info("Processing as a context query (document-specific)")
                response = self.query_agent.process_query(
                    user_query["message"],
                    session_name,
                    user_query["input_language"],
                    user_query["output_language"],
                    user_query["filenames"],
                    user_query["hascsvxl"],
                    user_query["mode"],
                    is_trial=False
                )
        except Exception as e:
            logger.exception(f'Error processing query with agent: {e}')
            return {'message': 'Error generating response'}, 500

        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return response, 200
    
    def process_demo_query(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Process a demo query for public transport information.
        
        Args:
            data: Query data including message
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        start_time = time.time()
        logger.info("Processing demo query")
        
        # Extract query parameters
        try:
            user_query = data.get('message')
            if not user_query:
                logger.warning("Empty demo query")
                return {'message': 'Query is missing'}, 400
        except Exception as e:
            logger.exception(f'Invalid request format: {e}')
            return {'message': str(e)}, 400
        
        # Generate response using LLM
        try:
            from controllers.ask import get_demo_response
            logger.info(f'Demo query: {user_query}')
            session_name = "user20250317t114507"  # Fixed demo session
            llm_response = get_demo_response(user_query, session_name)
        except Exception as e:
            logger.exception(f'Error generating demo response: {e}')
            return {'message': 'Error generating response from LLM'}, 500

        logger.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
        return {"answer": llm_response}, 200
    
    def _extract_query_parameters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate query parameters from request data.
        
        Args:
            data: Raw request data
            
        Returns:
            Dict with extracted parameters
        
        Raises:
            ValueError: If required parameters are missing
        """
        user_query = data.get('message')
        if not user_query:
            raise ValueError("Query message is missing")
            
        input_language = int(data.get('inputLanguage', 23))
        output_language = int(data.get('outputLanguage', 23))
        context = data.get('context', False)
        hascsvxl = data.get('hasCsvOrXlsx', False)
        mode = data.get('mode', 'default')
        filenames = data.get('filenames', [])
        
        return {
            "message": user_query,
            "input_language": input_language,
            "output_language": output_language,
            "context": context,
            "hascsvxl": hascsvxl,
            "mode": mode,
            "filenames": filenames
        }

# Create a singleton instance
_query_service = None

def get_query_service() -> QueryService:
    """
    Get the query service singleton instance.
    
    Returns:
        QueryService: The query service instance
    """
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service