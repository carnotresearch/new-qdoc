"""
Creative reasoning service that orchestrates the entire reasoning pipeline.

This module coordinates strategy creation, search execution, and result synthesis
to provide enhanced query processing for creative mode.
"""

import logging
import time
from typing import Dict, Any, Optional

from app.services.reasoning_strategy_service import get_reasoning_strategy_service
from app.services.search_execution_service import get_search_execution_service
from app.services.result_synthesis_service import get_result_synthesis_service
from app.services.context_provider_service import get_context_provider_service

# Configure logging
logger = logging.getLogger(__name__)

class CreativeReasoningService:
    """Main service for creative mode query processing with chain-of-thought reasoning."""
    
    def __init__(self):
        """Initialize the creative reasoning service."""
        logger.info("Initializing creative reasoning service")
        self.strategy_service = get_reasoning_strategy_service()
        self.execution_service = get_search_execution_service()
        self.synthesis_service = get_result_synthesis_service()
        self.context_service = get_context_provider_service()
        
        # Configuration
        self.max_processing_time = 60  # seconds
        self.enable_reasoning_steps_in_response = True
    
    def process_creative_query(self, user_query: str,
                             user_session: str,
                             available_resources: Dict[str, bool],
                             input_language: int = 23,
                             output_language: int = 23,
                             filenames: Optional[list] = None) -> Dict[str, Any]:
        """
        Process a query using creative reasoning mode.
        
        Args:
            user_query: The user's question
            user_session: User's session identifier
            available_resources: Dict indicating what resources are available
            input_language: Input language code
            output_language: Output language code
            filenames: Optional list of specific filenames to focus on
            
        Returns:
            Enhanced response with reasoning steps
        """
        start_time = time.time()
        logger.info(f"Processing creative query: {user_query}")
        
        try:
            # Step 1: Analyze query and create strategy
            logger.info("Step 1: Creating reasoning strategy")
            strategy = self._create_reasoning_strategy(
                user_query, available_resources, filenames
            )
            
            # Step 2: Convert strategy to executable instructions
            logger.info("Step 2: Converting strategy to search instructions")
            search_instructions = self.strategy_service.convert_to_search_instructions(strategy)
            
            if not search_instructions:
                return self._create_fallback_response(user_query, output_language)
            
            # Step 3: Execute search strategy
            logger.info(f"Step 3: Executing {len(search_instructions)} search instructions")
            search_results = self.execution_service.execute_search_strategy(
                search_instructions, user_session, user_query
            )
            
            # Step 4: Filter and rank results
            logger.info("Step 4: Filtering and ranking results")
            filtered_results = self.execution_service.filter_and_rank_results(
                search_results, user_query
            )
            
            # Step 5: Synthesize results
            logger.info("Step 5: Synthesizing results")
            language_name = self._get_language_name(output_language)
            synthesis_result = self.synthesis_service.synthesize_results(
                filtered_results, user_query, strategy, language_name
            )
            
            # Step 6: Format response
            logger.info("Step 6: Formatting final response")
            response = self._format_creative_response(
                synthesis_result, strategy, search_results, filtered_results
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Creative query processing completed in {processing_time:.2f} seconds")
            
            # Add processing metadata
            response["processing_metadata"] = {
                "processing_time": processing_time,
                "total_searches": len(search_instructions),
                "successful_searches": len(filtered_results),
                "strategy_reasoning": strategy.reasoning
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in creative query processing: {e}")
            processing_time = time.time() - start_time
            
            return self._create_error_response(user_query, str(e), processing_time, output_language)
    
    def _create_reasoning_strategy(self, user_query: str,
                                 available_resources: Dict[str, bool],
                                 filenames: Optional[list] = None) -> Any:
        """Create a reasoning strategy for the query."""
        # Gather context information
        context_info = {
            'query_complexity': self.strategy_service.estimate_complexity(user_query),
            'filenames': filenames
        }
        
        # If specific filenames are provided, add that information
        if filenames:
            context_info['focused_search'] = True
            context_info['file_count'] = len(filenames)
        
        return self.strategy_service.analyze_query_and_create_strategy(
            user_query, available_resources, context_info
        )
    
    def _format_creative_response(self, synthesis_result: Any,
                                strategy: Any,
                                all_search_results: list,
                                filtered_results: list) -> Dict[str, Any]:
        """Format the response for creative mode."""
        
        # Convert to legacy format for compatibility
        base_response = self.synthesis_service.create_legacy_format_response(synthesis_result)
        
        # Add creative mode enhancements
        if self.enable_reasoning_steps_in_response:
            base_response["creative_reasoning"] = {
                "strategy_used": strategy.reasoning,
                "query_breakdown": strategy.query_breakdown,
                "reasoning_steps": synthesis_result.reasoning_steps,
                "confidence_level": synthesis_result.confidence_level,
                "sources_analyzed": len(all_search_results),
                "sources_used": len(filtered_results),
                "synthesis_approach": strategy.synthesis_approach,
                "answer_completeness": getattr(synthesis_result, 'answer_completeness', 'fully answered'),
                "source_reliability": getattr(synthesis_result, 'source_reliability', 'good')
            }
            
            # Add search execution summary
            base_response["search_execution_summary"] = {
                "total_searches_planned": len(strategy.searches),
                "successful_searches": len([r for r in all_search_results if r.success]),
                "search_types_used": list(set([r.metadata.get('search_type', 'unknown') for r in all_search_results if r.success])),
                "multi_step_breakdown": [
                    f"Sub-question {i+1}: {q}" for i, q in enumerate(strategy.query_breakdown)
                ]
            }
        
        # Enhanced follow-up questions
        base_response["questions"] = synthesis_result.follow_up_questions
        
        return base_response
    
    def _create_fallback_response(self, user_query: str, output_language: int) -> Dict[str, Any]:
        """Create a fallback response when strategy creation fails."""
        language_name = self._get_language_name(output_language)
        
        fallback_message = (
            "I was unable to create an effective search strategy for your question. "
            "This might be because the question is very complex or the available resources "
            "are limited. Please try rephrasing your question or breaking it down into smaller parts."
        )
        
        if language_name != 'English':
            fallback_message = f"[Response in {language_name}] {fallback_message}"
        
        return {
            "answer": fallback_message,
            "fileName": "",
            "pageNo": 0,
            "sources": [],
            "questions": [
                "Could you break down your question into smaller parts?",
                "Are there specific aspects of this topic you'd like me to focus on?"
            ],
            "creative_reasoning": {
                "strategy_used": "fallback",
                "reasoning_steps": ["Unable to create search strategy", "Providing fallback response"],
                "confidence_level": "low"
            }
        }
    
    def _create_error_response(self, user_query: str, error_message: str,
                             processing_time: float, output_language: int) -> Dict[str, Any]:
        """Create an error response."""
        language_name = self._get_language_name(output_language)
        
        error_response = (
            "I encountered an error while processing your question using creative reasoning. "
            "Please try again or use the standard mode for this query."
        )
        
        if language_name != 'English':
            error_response = f"[Response in {language_name}] {error_response}"
        
        return {
            "answer": error_response,
            "fileName": "",
            "pageNo": 0,
            "sources": [],
            "questions": [],
            "creative_reasoning": {
                "strategy_used": "error_handling",
                "reasoning_steps": ["Error occurred during processing"],
                "confidence_level": "low",
                "error_details": error_message
            },
            "processing_metadata": {
                "processing_time": processing_time,
                "status": "error"
            }
        }
    
    def _get_language_name(self, language_code: int) -> str:
        """Get the language name from its code."""
        languages = {
            1: "Hindi", 2: "Gom", 3: "Kannada", 4: "Dogri", 5: "Bodo",
            6: "Urdu", 7: "Tamil", 8: "Kashmiri", 9: "Assamese", 10: "Bengali",
            11: "Marathi", 12: "Sindhi", 13: "Maithili", 14: "Punjabi", 15: "Malayalam",
            16: "Manipuri", 17: "Telugu", 18: "Sanskrit", 19: "Nepali", 20: "Santali",
            21: "Gujarati", 22: "Odia", 23: "English"
        }
        return languages.get(language_code, 'English')
    
    def should_use_creative_mode(self, mode: str, query_complexity: Optional[str] = None) -> bool:
        """
        Determine if creative mode should be used.
        
        Args:
            mode: The mode parameter from the request
            query_complexity: Optional complexity assessment
            
        Returns:
            True if creative mode should be used
        """
        return mode == 'creative'
    
    def get_creative_mode_info(self) -> Dict[str, Any]:
        """Get information about creative mode capabilities."""
        return {
            "name": "Creative Reasoning Mode",
            "description": "Enhanced query processing with chain-of-thought reasoning and multi-step analysis",
            "features": [
                "Multi-step query decomposition (up to 5 sub-questions)",
                "Intelligent search type selection (documents, SQL, hybrid)",
                "Advanced result synthesis with reasoning transparency",
                "Cross-source validation and pattern recognition",
                "Enhanced source utilization and dependency management"
            ],
            "best_for": [
                "Complex analytical questions requiring multi-step reasoning",
                "Comparative analysis across different data sources",
                "Research queries needing comprehensive information gathering",
                "Questions requiring both textual and numerical insights",
                "Exploratory analysis with pattern recognition"
            ],
            "processing_time": "10-60 seconds depending on complexity",
            "max_search_steps": 5,
            "reasoning_transparency": "Full reasoning steps provided to user",
            "example_breakdown": {
                "query": "How does our Q3 performance compare to competitors and what drove the differences?",
                "sub_questions": [
                    "What were our Q3 financial metrics?",
                    "What were competitor Q3 performance metrics?", 
                    "What market factors influenced Q3 performance?",
                    "What internal factors affected our performance?",
                    "How do these factors explain the performance differences?"
                ]
            }
        }

# Create a singleton instance
_creative_reasoning_service = None

def get_creative_reasoning_service() -> CreativeReasoningService:
    """
    Get the creative reasoning service singleton instance.
    
    Returns:
        CreativeReasoningService: The creative reasoning service instance
    """
    global _creative_reasoning_service
    if _creative_reasoning_service is None:
        _creative_reasoning_service = CreativeReasoningService()
    return _creative_reasoning_service