"""
Updated Query agent service with creative reasoning integration.

This module implements both standard and creative query processing modes,
automatically selecting the appropriate approach based on user preferences.
"""

import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple

from app.services.query_intent_service import get_query_intent_service, QueryIntent
from app.services.context_provider_service import get_context_provider_service
from app.services.response_generator_service import get_response_generator_service
from utils.extractText import clean_filename

# Import creative reasoning service
try:
    from app.services.creative_reasoning_service import get_creative_reasoning_service
    CREATIVE_MODE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative reasoning service not available: {e}")
    CREATIVE_MODE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class QueryAgentService:
    """Enhanced agentic service for processing user queries with creative mode support."""
    
    def __init__(self):
        """Initialize the query agent service."""
        logger.info("Initializing enhanced query agent service")
        self.intent_service = get_query_intent_service()
        self.context_service = get_context_provider_service()
        self.response_service = get_response_generator_service()
        
        # Initialize creative reasoning service if available
        if CREATIVE_MODE_AVAILABLE:
            try:
                self.creative_service = get_creative_reasoning_service()
                logger.info("Creative reasoning service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize creative reasoning service: {e}")
                self.creative_service = None
        else:
            self.creative_service = None
            logger.info("Creative reasoning service not available")
    
    def process_no_context_query(self, user_query: str, user_email: str,
                          input_language: int = 23, output_language: int = 23) -> Dict[str, Any]:
        """
        Process a query when no document context is available.
        Note: Creative mode is not applicable for no-context queries.
        
        Args:
            user_query: The user's query text
            user_email: User's email for personalization
            input_language: Input language code
            output_language: Output language code
            
        Returns:
            Response data
        """
        # Get language name for response
        language = self._get_language(output_language)
        logger.info(f'Processing no-context query: "{user_query}" for user {user_email}')
        
        # Get list of available sessions for this user
        available_sessions = self._get_available_sessions(user_email)
        
        # Generate a response that guides the user to upload files or select a session
        return self._generate_no_context_response(user_query, user_email, available_sessions, language)
        
    def process_query(self, user_query: str, user_session: str, 
                    input_language: int = 23, output_language: int = 23,
                    filenames: Optional[List[str]] = None,
                    has_csvxl: bool = False, mode: str = 'default',
                    is_trial: bool = False) -> Dict[str, Any]:
        """
        Process a user query with support for both standard and creative modes.
        
        Args:
            user_query: The user's query text
            user_session: User's session identifier
            input_language: Input language code
            output_language: Output language code
            filenames: List of filenames to filter results by
            has_csvxl: Flag for CSV/Excel data
            mode: Processing mode ('default' or 'creative')
            is_trial: Whether this is a trial user
            
        Returns:
            Response data
        """
        start_time = time.time()
        logger.info(f'Processing query: "{user_query}" for session {user_session}, mode: {mode}')
        
        # Get language name for response
        language = self._get_language(output_language)
        
        # Check available resources for this session
        resources = self.context_service.check_resources_exist(user_session)
        logger.info(f"Available resources: {resources}")
        
        # Always update has_csvxl based on actual resource availability
        has_csvxl = has_csvxl or resources.get('has_data_tables', False)
        
        # Determine if creative mode should be used
        if self._should_use_creative_mode(mode, user_query, resources, is_trial):
            logger.info("Using creative reasoning mode")
            return self._process_creative_query(
                user_query, user_session, resources, 
                input_language, output_language, filenames
            )
        
        # Use standard processing mode
        logger.info("Using standard processing mode")
        return self._process_standard_query(
            user_query, user_session, resources, language, 
            filenames, has_csvxl
        )
    
    def _should_use_creative_mode(self, mode: str, user_query: str, 
                                resources: Dict[str, bool], is_trial: bool) -> bool:
        """
        Determine if creative mode should be used for this query.
        
        Args:
            mode: The mode parameter from request
            user_query: The user's query
            resources: Available resources
            is_trial: Whether this is a trial user
            
        Returns:
            True if creative mode should be used
        """
        # Creative mode is explicitly requested
        if mode != 'creative':
            return False
        
        # Check if creative service is available
        if not CREATIVE_MODE_AVAILABLE or self.creative_service is None:
            logger.warning("Creative mode requested but service not available")
            return False
        
        # Don't use creative mode for trial users (could be resource intensive)
        if is_trial:
            logger.info("Creative mode disabled for trial users")
            return False
        
        # Need some resources available for creative mode to be useful
        if not any(resources.values()):
            logger.info("No resources available, skipping creative mode")
            return False
        
        # Check if query is substantial enough for creative mode
        if len(user_query.split()) < 3:
            logger.info("Query too simple for creative mode")
            return False
        
        return True
    
    def _process_creative_query(self, user_query: str, user_session: str,
                              resources: Dict[str, bool],
                              input_language: int, output_language: int,
                              filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process query using creative reasoning mode."""
        try:
            response = self.creative_service.process_creative_query(
                user_query=user_query,
                user_session=user_session,
                available_resources=resources,
                input_language=input_language,
                output_language=output_language,
                filenames=filenames
            )
            
            # Save query history for creative mode responses too
            self.response_service.save_query_history(
                user_session, user_query, response.get('answer', '')
            )
            
            return response
            
        except Exception as e:
            logger.error(f'Error in creative mode processing: {e}')
            
            # Fallback to standard processing
            logger.info("Falling back to standard processing mode")
            language = self._get_language(output_language)
            return self._process_standard_query(
                user_query, user_session, resources, language, filenames, 
                resources.get('has_data_tables', False)
            )
    
    def _process_standard_query(self, user_query: str, user_session: str,
                              resources: Dict[str, bool], language: Optional[str],
                              filenames: Optional[List[str]], has_csvxl: bool) -> Dict[str, Any]:
        """Process query using standard mode (existing logic)."""
        
        # Classify query intent
        intent, confidence = self.intent_service.classify_intent(
            user_query, 
            has_documents=resources.get('has_documents', False),
            has_data_tables=has_csvxl
        )
        logger.info(f'Query intent classification: {intent.name} with confidence {confidence}')
        
        # For general chat queries when documents are available, use document-aware chat
        if intent == QueryIntent.GENERAL_CHAT and resources.get('has_documents', False):
            return self._process_document_aware_chat(user_query, user_session, language, filenames)
        
        # Process based on query intent
        if intent == QueryIntent.GENERAL_CHAT:
            return self._process_general_chat(user_query, language)
            
        elif intent == QueryIntent.SUMMARY:
            return self._process_summary_query(user_query, user_session, language, filenames)
            
        elif intent == QueryIntent.DOCUMENT:
            return self._process_document_query(user_query, user_session, language)
            
        elif intent == QueryIntent.DATA_QUERY:
            return self._process_data_query(user_query, user_session, language)
            
        elif intent == QueryIntent.HYBRID:
            return self._process_hybrid_query(user_query, user_session, language)
            
        # Fallback to general response if no specific handler
        return self._process_general_chat(user_query, language)
    
    def _process_general_chat(self, user_query: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Process a general chat query."""
        return self.response_service.generate_general_chat_response(user_query, language)
        
    def _process_summary_query(self, user_query: str, user_session: str, 
                             language: Optional[str] = None,
                             filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a summary query."""
        # Convert filenames to folder names if provided
        folder_names = []
        if filenames and len(filenames) > 0:
            from werkzeug.utils import secure_filename
            import os
            for filename in filenames:
                # Convert filename to secure folder name format
                safe_folder_name = secure_filename(os.path.splitext(filename)[0])
                if safe_folder_name:
                    folder_names.append(safe_folder_name)
                    
        logger.info(f"Using folder names for summary: {folder_names}")
                    
        # Get summary context
        try:
            summary = self.context_service.get_summary_context(
                user_session, user_query, language, folder_names
            )
            return self.response_service.generate_summary_response(summary)
        except Exception as e:
            logger.error(f'Error generating summary: {e}')
            return {
                "answer": "I encountered an error while generating the summary. Please try again later.",
                "sources": [],
                "questions": []
            }
            
    def _process_document_query(self, user_query: str, user_session: str, 
                              language: Optional[str] = None) -> Dict[str, Any]:
        """Process a document query."""
        try:
            # Get document context
            context, file_name, page_no = self.context_service.get_document_context(
                user_session, user_query
            )
            
            # If no context found, return a message about no relevant documents
            if not context:
                return {
                    "answer": "I couldn't find any relevant information in your documents to answer this question. Could you try rephrasing your query or asking about another topic?",
                }
                
            # Generate response
            response = self.response_service.generate_document_response(
                user_query, context, file_name, page_no, language
            )
            
            # Save query history
            self.response_service.save_query_history(
                user_session, user_query, response.get('answer', '')
            )
            
            return response
        except Exception as e:
            logger.error(f'Error processing document query: {e}')
            return {
                "answer": f"I encountered an error while processing your document: {str(e)}. Please try a different question or contact support if the issue persists.",
            }
        
    def _process_data_query(self, user_query: str, user_session: str, 
                          language: Optional[str] = None) -> Dict[str, Any]:
        """Process a data query."""
        # Get SQL context
        sql_context = self.context_service.get_data_context(user_session, user_query)
        
        # If no SQL context found, return a message about no relevant data
        if not sql_context:
            return {
                "answer": "I couldn't find any relevant data in your spreadsheets or CSV files to answer this question. Could you try rephrasing your query or asking about another topic?",
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
            
        # Generate response
        response = self.response_service.generate_data_response(
            user_query, sql_context, language
        )
        
        # Save query history
        self.response_service.save_query_history(
            user_session, user_query, response.get('answer', '')
        )
        
        return response
        
    def _process_hybrid_query(self, user_query: str, user_session: str, 
                            language: Optional[str] = None) -> Dict[str, Any]:
        """Process a hybrid query needing both document and data context."""
        # Get both document and SQL contexts
        document_context, file_name, page_no = self.context_service.get_document_context(
            user_session, user_query
        )
        
        sql_context = self.context_service.get_data_context(user_session, user_query)
        
        # If neither context found, return a message about no relevant information
        if not document_context and not sql_context:
            return {
                "answer": "I couldn't find any relevant information in your documents or data to answer this question. Could you try rephrasing your query or asking about another topic?",
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
            
        # Generate response
        response = self.response_service.generate_hybrid_response(
            user_query, document_context, sql_context, file_name, page_no, language
        )
        
        # Save query history
        self.response_service.save_query_history(
            user_session, user_query, response.get('answer', '')
        )
        
        return response
        
    def _process_document_aware_chat(self, user_query: str, user_session: str,
                               language: Optional[str] = None,
                               filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a general chat query when documents are available."""
        logger.info(f"Processing document-aware general chat: {user_query}")
        
        # Get basic information about the available documents
        documents_info = self._get_documents_info(user_session, filenames)
        
        # Generate response
        response = self.response_service.generate_document_aware_chat_response(
            user_query, documents_info, language
        )
        
        # Save query history
        self.response_service.save_query_history(
            user_session, user_query, response.get('answer', '')
        )
        
        return response
    
    def _get_available_sessions(self, user_email: str) -> List[str]:
        """Get a list of available sessions for a user."""
        try:
            import os
            import glob
            
            # Pattern to match user sessions - adjust as needed based on your storage structure
            pattern = f'users/{user_email}*'
            sessions = glob.glob(pattern)
            
            # Extract session identifiers
            return [os.path.basename(s) for s in sessions]
        except Exception as e:
            logger.error(f'Error getting available sessions: {e}')
            return []
    
    def _generate_no_context_response(self, user_query: str, user_email: str,
                                   available_sessions: List[str], language: str) -> Dict[str, Any]:
        """Generate a response for a user who hasn't selected a document context."""
        prompt = f"""You are a chat bot called icarKno created by Carnot Research Pvt Ltd, which answers queries related to documents.
        
        The user is currently in the main chat interface but hasn't selected any specific documents or knowledge containers.
        
        Answer the user's question in a helpful, conversational tone. Since no documents are currently selected, you should:
        
        1. Respond naturally to general questions
        2. For document-specific questions, politely remind them to select a knowledge container from the left sidebar or upload new files
        3. Mention that they have {len(available_sessions)} existing knowledge containers they can choose from (if they have any)
        4. Be specific about how to upload new files (click the "New Container" button) or select existing containers (from the left menu)
        
        User's question: {user_query}
        
        Keep your response concise, helpful and focused on guiding the user without being overly repetitive or robotic.
        """
        
        # Add language in prompt if not English
        if language and language != 'English':
            prompt = prompt + f"\n\nAnswer in the user's preferred language - {language}."
            
        # Generate response
        try:
            llm_response = self.response_service.llm.invoke(prompt)
            response = str(llm_response.content)
            
            return {
                "answer": response,
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
        except Exception as e:
            logger.error(f'Error generating no-context response: {e}')
            return {
                "answer": "I'm here to help you explore your documents. Please select a knowledge container from the left sidebar or upload new documents to get started.",
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
    
    def _get_documents_info(self, user_session: str, 
                          filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get basic information about available documents."""
        try:
            # Get information about available files
            file_count = 0
            file_types = set()
            topics = "various"
            
            # Check files directory
            files_dir = os.path.join('users', user_session, 'files')
            if os.path.exists(files_dir):
                import glob
                
                # If filenames are provided, filter to those specific files
                if filenames and len(filenames) > 0:
                    for filename in filenames:
                        # Extract file extension
                        ext = os.path.splitext(filename)[1].lower()
                        if ext:
                            file_types.add(ext[1:])  # Remove the dot
                    file_count = len(filenames)
                else:
                    # Count all files in the directory
                    file_dirs = glob.glob(os.path.join(files_dir, '*'))
                    file_count = len(file_dirs)
                    
                    # Get file types
                    for file_dir in file_dirs:
                        metadata_path = os.path.join(file_dir, 'metadata.json')
                        if os.path.exists(metadata_path):
                            import json
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                filename = metadata.get('filename', '')
                                if filename:
                                    ext = os.path.splitext(filename)[1].lower()
                                    if ext:
                                        file_types.add(ext[1:])  # Remove the dot
            
            # Check for legacy content.txt
            legacy_content = os.path.join('users', user_session, 'content.txt')
            if os.path.exists(legacy_content):
                file_count += 1
                file_types.add('txt')
            
            # Check for SQL/Excel files
            sheet_metadata_path = os.path.join('users', user_session, "files", "sheet_metadata.json")
            if os.path.exists(sheet_metadata_path):
                import json
                with open(sheet_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    for filename in metadata:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext:
                            file_types.add(ext[1:])  # Remove the dot
                        file_count += 1
            
            return {
                "file_count": file_count,
                "file_types": list(file_types) if file_types else ["document"],
                "topics": topics
            }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {
                "file_count": "multiple",
                "file_types": ["document"],
                "topics": "various"
            }
            
    def _get_language(self, language_code: int) -> str:
        """Get the language name from its code."""
        languages = {
            1: "Hindi", 2: "Gom", 3: "Kannada", 4: "Dogri", 5: "Bodo",
            6: "Urdu", 7: "Tamil", 8: "Kashmiri", 9: "Assamese", 10: "Bengali",
            11: "Marathi", 12: "Sindhi", 13: "Maithili", 14: "Punjabi", 15: "Malayalam",
            16: "Manipuri", 17: "Telugu", 18: "Sanskrit", 19: "Nepali", 20: "Santali",
            21: "Gujarati", 22: "Odia", 23: "English"
        }
        return languages.get(language_code, 'English')
    
    def get_supported_modes(self) -> Dict[str, Any]:
        """Get information about supported query processing modes."""
        modes = {
            "default": {
                "name": "Standard Mode",
                "description": "Fast, efficient query processing with intent classification",
                "processing_time": "1-5 seconds",
                "best_for": ["Simple questions", "Quick lookups", "Direct answers"]
            }
        }
        
        if CREATIVE_MODE_AVAILABLE and self.creative_service is not None:
            modes["creative"] = {
                "name": "Creative Reasoning Mode", 
                "description": "Enhanced processing with chain-of-thought reasoning",
                "processing_time": "10-60 seconds",
                "best_for": ["Complex analysis", "Multi-step reasoning", "Research queries"],
                "features": self.creative_service.get_creative_mode_info()
            }
        
        return modes

# Create a singleton instance
_query_agent_service = None

def get_query_agent_service() -> QueryAgentService:
    """
    Get the enhanced query agent service singleton instance.
    
    Returns:
        QueryAgentService: The enhanced query agent service instance
    """
    global _query_agent_service
    if _query_agent_service is None:
        _query_agent_service = QueryAgentService()
    return _query_agent_service