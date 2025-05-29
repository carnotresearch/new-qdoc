"""
Result synthesis service for combining multiple search results into coherent answers.

This module handles the synthesis of search results from multiple sources,
applying reasoning and creating comprehensive responses.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from app.services.search_execution_service import SearchResult
from app.services.reasoning_strategy_service import SearchStrategy
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    """Result of the synthesis process."""
    answer: str
    reasoning_steps: List[str]
    sources_used: List[Dict[str, Any]]
    confidence_level: str
    follow_up_questions: List[str]
    synthesis_metadata: Dict[str, Any]
    source_reliability: Optional[str] = None
    answer_completeness: Optional[str] = None

class ResultSynthesisService:
    """Service for synthesizing multiple search results into coherent answers."""
    
    def __init__(self):
        """Initialize the result synthesis service."""
        logger.info("Initializing result synthesis service")
        self.openai_api_key = Config.OPENAI_API_KEY
        self.llm = ChatOpenAI(model="gpt-4o", api_key=self.openai_api_key, temperature=0.2)
        self.max_sources_in_response = 10
    
    def synthesize_results(self, search_results: List[SearchResult],
                          original_query: str,
                          strategy: SearchStrategy,
                          language: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize multiple search results into a comprehensive answer.
        
        Args:
            search_results: List of search results from execution
            original_query: The original user query
            strategy: The search strategy that was used
            language: Preferred response language
            
        Returns:
            SynthesisResult with comprehensive answer and metadata
        """
        logger.info(f"Synthesizing {len(search_results)} search results")
        
        # Filter and prepare results
        successful_results = [r for r in search_results if r.success and r.content.strip()]
        
        if not successful_results:
            return self._create_no_results_response(original_query, language)
        
        # Create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(
            successful_results, original_query, strategy, language
        )
        
        try:
            # Generate synthesis
            logger.info(f"Sending synthesis prompt to LLM (approx {len(synthesis_prompt)} characters)")
            response = self.llm.invoke(synthesis_prompt)
            logger.info(f"LLM Synthesis Response: {response.content}")
            
            synthesis_response = self._parse_synthesis_response(response.content)
            logger.info(f"Parsed synthesis response keys: {list(synthesis_response.keys())}")
            
            # Create final result
            return SynthesisResult(
                answer=synthesis_response.get('answer', ''),
                reasoning_steps=synthesis_response.get('reasoning_steps', []),
                sources_used=self._extract_sources_used(successful_results),
                confidence_level=synthesis_response.get('confidence_level', 'medium'),
                follow_up_questions=synthesis_response.get('follow_up_questions', []),
                synthesis_metadata={
                    'total_sources': len(successful_results),
                    'strategy_used': strategy.reasoning,
                    'synthesis_approach': strategy.synthesis_approach
                },
                source_reliability=synthesis_response.get('source_reliability', 'moderate'),
                answer_completeness=synthesis_response.get('answer_completeness', 'partially answered')
            )
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return self._create_fallback_synthesis(successful_results, original_query, language)
    
    def _create_synthesis_prompt(self, results: List[SearchResult],
                               original_query: str,
                               strategy: SearchStrategy,
                               language: Optional[str] = None) -> str:
        """Create a comprehensive prompt for result synthesis."""
        
        # Prepare search results for the prompt
        results_text = self._format_results_for_prompt(results)
        
        # Create the main synthesis prompt
        prompt = f"""You are an expert research analyst tasked with synthesizing multiple search results into a comprehensive, accurate answer.

ORIGINAL QUERY: {original_query}

RESEARCH STRATEGY USED:
{strategy.reasoning}

SYNTHESIS APPROACH:
{strategy.synthesis_approach}

SEARCH RESULTS:
{results_text}

YOUR TASK:
Analyze all the search results and create a comprehensive answer that:

1. **REASONING CHAIN**: Show your step-by-step reasoning process
2. **COMPREHENSIVE ANSWER**: Synthesize information from multiple sources
3. **SOURCE INTEGRATION**: Seamlessly integrate information while citing sources
4. **CONFIDENCE ASSESSMENT**: Evaluate the reliability of your answer
5. **FOLLOW-UP QUESTIONS**: Suggest relevant follow-up questions

RESPONSE FORMAT:
Provide your response as a JSON object with this exact structure:

{{
    "reasoning_steps": [
        "Step 1: Analysis of available information...",
        "Step 2: Cross-referencing sources...",
        "Step 3: Identifying patterns and insights...",
        "Step 4: Synthesis and conclusion..."
    ],
    "answer": "Your comprehensive answer with inline source citations [Source X]",
    "confidence_level": "high|medium|low",
    "follow_up_questions": [
        "Relevant follow-up question 1?",
        "Relevant follow-up question 2?",
        "Relevant follow-up question 3?"
    ],
    "source_reliability": "Assessment of how reliable the sources are"
}}

CITATION RULES:
- Use [Source X] format where X is the source number from the search results
- Cite specific sources for specific claims
- Don't cite sources for general knowledge or logical inferences
- If information conflicts between sources, acknowledge the conflict

QUALITY REQUIREMENTS:
- Be thorough but concise
- Acknowledge limitations or gaps in the data
- If sources conflict, explain the discrepancy
- Maintain logical flow in your reasoning
- Ensure answer directly addresses the original query"""

        # Add language specification if provided
        if language and language != 'English':
            prompt += f"\n\nLANGUAGE: Provide your answer in {language}."
        
        prompt += "\n\nGenerate ONLY the JSON response. Do not include any other text."
        
        return prompt
    
    def _format_results_for_prompt(self, results: List[SearchResult]) -> str:
        """Format search results for inclusion in the prompt."""
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            search_type = result.metadata.get('search_type', 'unknown')
            
            # Create source header
            source_header = f"[Source {i}] ({search_type.upper()})"
            
            # Add source info if available
            if result.source_info:
                if result.source_info.get('file_name'):
                    source_header += f" - {result.source_info['file_name']}"
                if result.source_info.get('page_no'):
                    source_header += f", Page {result.source_info['page_no']}"
            
            # Format content (truncate if too long)
            content = result.content
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            
            formatted_results.append(f"{source_header}:\n{content}\n")
        
        return "\n".join(formatted_results)
    
    def _parse_synthesis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse the JSON response from the synthesis LLM."""
        try:
            # Clean up the response
            cleaned_response = response_content.strip()
            
            # Remove any markdown code block markers
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            parsed = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ['reasoning_steps', 'answer', 'confidence_level', 'follow_up_questions']
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = self._get_default_value(field)
            
            # Handle optional new fields
            if 'source_reliability' not in parsed:
                parsed['source_reliability'] = 'moderate'
            if 'answer_completeness' not in parsed:
                parsed['answer_completeness'] = 'partially answered'
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse synthesis response as JSON: {e}")
            return self._extract_fallback_content(response_content)
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields."""
        defaults = {
            'reasoning_steps': [
                "Step 1 - Information Analysis: Analyzed available information from sources",
                "Step 2 - Pattern Recognition: Identified key patterns and themes", 
                "Step 3 - Cross-Validation: Cross-referenced information across sources",
                "Step 4 - Gap Analysis: Noted any limitations in available data",
                "Step 5 - Synthesis: Combined information to formulate comprehensive answer"
            ],
            'answer': "Based on the available information...",
            'confidence_level': 'medium',
            'follow_up_questions': [],
            'source_reliability': 'moderate',
            'answer_completeness': 'partially answered'
        }
        return defaults.get(field, "")
    
    def _extract_fallback_content(self, response_content: str) -> Dict[str, Any]:
        """Extract content when JSON parsing fails."""
        # Try to extract answer from the response text
        answer_match = re.search(r'"answer":\s*"([^"]+)"', response_content)
        answer = answer_match.group(1) if answer_match else response_content[:1000]
        
        return {
            'reasoning_steps': ["Unable to parse detailed reasoning"],
            'answer': answer,
            'confidence_level': 'low',
            'follow_up_questions': [],
            'source_reliability': 'unknown'
        }
    
    def _extract_sources_used(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract source information from successful results."""
        sources = []
        
        for i, result in enumerate(results, 1):
            source_info = {
                'source_number': i,
                'search_type': result.metadata.get('search_type', 'unknown'),
                'query_used': result.metadata.get('query', ''),
                'content_length': len(result.content)
            }
            
            # Add file information if available
            if result.source_info:
                source_info.update(result.source_info)
            
            sources.append(source_info)
        
        return sources
    
    def _create_no_results_response(self, original_query: str, 
                                  language: Optional[str] = None) -> SynthesisResult:
        """Create a response when no search results are available."""
        no_results_message = (
            "I couldn't find any relevant information in your documents or data to answer this question. "
            "This could be because the information isn't available in your uploaded files, or the question "
            "might need to be rephrased to better match the available content."
        )
        
        if language and language != 'English':
            # In a real implementation, you might want to translate this
            no_results_message = f"[Response in {language}] " + no_results_message
        
        return SynthesisResult(
            answer=no_results_message,
            reasoning_steps=["No relevant search results found", "Unable to provide answer based on available data"],
            sources_used=[],
            confidence_level='low',
            follow_up_questions=[
                "Could you try rephrasing your question?",
                "Do you have additional documents that might contain this information?"
            ],
            synthesis_metadata={
                'total_sources': 0,
                'strategy_used': 'fallback',
                'synthesis_approach': 'no_results_handling'
            }
        )
    
    def _create_fallback_synthesis(self, results: List[SearchResult],
                                 original_query: str,
                                 language: Optional[str] = None) -> SynthesisResult:
        """Create a fallback synthesis when the main synthesis fails."""
        logger.info("Creating fallback synthesis")
        
        # Simple concatenation of results with basic formatting
        combined_content = ""
        sources_used = []
        
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            combined_content += f"\n\nFrom Source {i}:\n{result.content[:800]}"
            
            sources_used.append({
                'source_number': i,
                'search_type': result.metadata.get('search_type', 'unknown'),
                'content_length': len(result.content)
            })
        
        fallback_answer = f"Based on the available information:{combined_content}"
        
        if language and language != 'English':
            fallback_answer = f"[Response in {language}] {fallback_answer}"
        
        return SynthesisResult(
            answer=fallback_answer,
            reasoning_steps=["Fallback synthesis due to processing error", "Combined available search results"],
            sources_used=sources_used,
            confidence_level='medium',
            follow_up_questions=[],
            synthesis_metadata={
                'total_sources': len(results),
                'strategy_used': 'fallback',
                'synthesis_approach': 'simple_concatenation'
            }
        )
    
    def create_legacy_format_response(self, synthesis_result: SynthesisResult,
                                    file_name: Optional[str] = None,
                                    page_no: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert synthesis result to the legacy response format for compatibility.
        
        Args:
            synthesis_result: The synthesis result
            file_name: Primary file name for compatibility
            page_no: Primary page number for compatibility
            
        Returns:
            Response in legacy format
        """
        # Extract source information for legacy format
        legacy_sources = []
        for source in synthesis_result.sources_used:
            if source.get('file_name') and source.get('page_no') is not None:
                legacy_sources.append({
                    "fileName": source['file_name'],
                    "pageNo": source['page_no']
                })
        
        return {
            "answer": synthesis_result.answer,
            "fileName": file_name or (legacy_sources[0]["fileName"] if legacy_sources else ""),
            "pageNo": page_no or (legacy_sources[0]["pageNo"] if legacy_sources else 0),
            "sources": legacy_sources,
            "questions": synthesis_result.follow_up_questions,
            "reasoning_metadata": {
                "reasoning_steps": synthesis_result.reasoning_steps,
                "confidence_level": synthesis_result.confidence_level,
                "synthesis_metadata": synthesis_result.synthesis_metadata
            }
        }

# Create a singleton instance
_result_synthesis_service = None

def get_result_synthesis_service() -> ResultSynthesisService:
    """
    Get the result synthesis service singleton instance.
    
    Returns:
        ResultSynthesisService: The result synthesis service instance
    """
    global _result_synthesis_service
    if _result_synthesis_service is None:
        _result_synthesis_service = ResultSynthesisService()
    return _result_synthesis_service