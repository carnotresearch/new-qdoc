"""
Search execution service for coordinating multiple search operations.

This module handles the execution of search strategies, managing dependencies,
and collecting results from various search types.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.services.reasoning_strategy_service import SearchInstruction, SearchType, SearchPriority
from app.services.context_provider_service import get_context_provider_service

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result from a single search operation."""
    instruction_index: int
    success: bool
    content: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    source_info: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionPlan:
    """Plan for executing search instructions with dependency management."""
    phases: List[List[int]]  # List of phases, each containing instruction indices
    total_instructions: int
    estimated_duration: float

class SearchExecutionService:
    """Service for executing search strategies with coordination and dependency management."""
    
    def __init__(self):
        """Initialize the search execution service."""
        logger.info("Initializing search execution service")
        self.context_service = get_context_provider_service()
        self.max_concurrent_searches = 3
        self.search_timeout = 30  # seconds
    
    def execute_search_strategy(self, instructions: List[SearchInstruction], 
                              user_session: str,
                              original_query: str) -> List[SearchResult]:
        """
        Execute a complete search strategy with dependency management.
        
        Args:
            instructions: List of search instructions to execute
            user_session: User's session identifier
            original_query: The original user query for context
            
        Returns:
            List of search results
        """
        logger.info(f"Executing search strategy with {len(instructions)} instructions")
        
        # Create execution plan
        execution_plan = self._create_execution_plan(instructions)
        logger.info(f"Created execution plan with {len(execution_plan.phases)} phases")
        
        # Execute searches phase by phase
        all_results = []
        context_accumulator = {}
        
        for phase_num, phase_indices in enumerate(execution_plan.phases):
            logger.info(f"Executing phase {phase_num + 1} with {len(phase_indices)} searches")
            
            phase_results = self._execute_phase(
                instructions, phase_indices, user_session, 
                original_query, context_accumulator
            )
            
            all_results.extend(phase_results)
            
            # Update context accumulator with results from this phase
            self._update_context_accumulator(phase_results, context_accumulator)
        
        logger.info(f"Completed search strategy execution with {len(all_results)} results")
        return all_results
    
    def _create_execution_plan(self, instructions: List[SearchInstruction]) -> ExecutionPlan:
        """Create an execution plan considering dependencies and priorities."""
        phases = []
        remaining_indices = set(range(len(instructions)))
        completed_indices = set()
        
        while remaining_indices:
            # Find instructions that can be executed in this phase
            current_phase = []
            
            for idx in list(remaining_indices):
                instruction = instructions[idx]
                
                # Check if all dependencies are completed
                if instruction.dependencies:
                    dependencies_met = all(dep in completed_indices for dep in instruction.dependencies)
                else:
                    dependencies_met = True
                
                if dependencies_met:
                    current_phase.append(idx)
            
            if not current_phase:
                # If no instructions can be executed, there might be circular dependencies
                logger.warning("Possible circular dependencies detected, executing remaining instructions")
                current_phase = list(remaining_indices)
            
            # Sort by priority within the phase
            current_phase.sort(key=lambda idx: self._get_priority_order(instructions[idx].priority))
            
            phases.append(current_phase)
            
            # Update completed and remaining sets
            completed_indices.update(current_phase)
            remaining_indices -= set(current_phase)
        
        return ExecutionPlan(
            phases=phases,
            total_instructions=len(instructions),
            estimated_duration=self._estimate_execution_time(instructions)
        )
    
    def _get_priority_order(self, priority: SearchPriority) -> int:
        """Convert priority to numerical order for sorting."""
        priority_order = {
            SearchPriority.HIGH: 0,
            SearchPriority.MEDIUM: 1,
            SearchPriority.LOW: 2
        }
        return priority_order.get(priority, 1)
    
    def _execute_phase(self, instructions: List[SearchInstruction],
                      phase_indices: List[int],
                      user_session: str,
                      original_query: str,
                      context_accumulator: Dict[str, Any]) -> List[SearchResult]:
        """Execute a single phase of searches."""
        results = []
        
        for idx in phase_indices:
            instruction = instructions[idx]
            logger.info(f"Executing search {idx}: {instruction.search_type.name}")
            
            try:
                result = self._execute_single_search(
                    instruction, idx, user_session, original_query, context_accumulator
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing search {idx}: {e}")
                results.append(SearchResult(
                    instruction_index=idx,
                    success=False,
                    content="",
                    metadata={},
                    error_message=str(e)
                ))
        
        return results
    
    def _execute_single_search(self, instruction: SearchInstruction,
                             instruction_index: int,
                             user_session: str,
                             original_query: str,
                             context_accumulator: Dict[str, Any]) -> SearchResult:
        """Execute a single search instruction."""
        
        # Enhance query with context if available
        enhanced_query = self._enhance_query_with_context(
            instruction.query, context_accumulator, instruction.instructions
        )
        
        try:
            if instruction.search_type == SearchType.DOCUMENT_VECTOR:
                return self._execute_document_search(
                    enhanced_query, user_session, instruction_index, 'vector'
                )
            
            elif instruction.search_type == SearchType.DOCUMENT_KEYWORD:
                return self._execute_document_search(
                    enhanced_query, user_session, instruction_index, 'keyword'
                )
            
            elif instruction.search_type == SearchType.SQL_QUERY:
                return self._execute_sql_search(
                    enhanced_query, user_session, instruction_index
                )
            
            elif instruction.search_type == SearchType.SUMMARY:
                return self._execute_summary_search(
                    enhanced_query, user_session, instruction_index
                )
            
            elif instruction.search_type == SearchType.HYBRID:
                return self._execute_hybrid_search(
                    enhanced_query, user_session, instruction_index
                )
            
            else:
                raise ValueError(f"Unknown search type: {instruction.search_type}")
                
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise
    
    def _extract_all_sources_from_context(self, context: str) -> List[Dict[str, Any]]:
        """Extract all source citations from context string."""
        import re
        sources = []
        
        # Pattern to match context format: (Source: filename, Page X)
        pattern = r'\(Source:\s*([^,]+),\s*Page\s*(\d+)\)'
        matches = re.findall(pattern, context)
        
        for filename, page in matches:
            sources.append({
                'file_name': filename.strip(),
                'page_no': int(page)
            })
        
        return sources

    def _execute_document_search(self, query: str, user_session: str, 
                               instruction_index: int, search_mode: str) -> SearchResult:
        """Execute document search (vector or keyword)."""
        try:
            context, file_name, page_no = self.context_service.get_document_context(
                user_session, query
            )
            
            if context:
                # Parse context to extract all source information
                source_info = self._extract_all_sources_from_context(context)
                
                # Primary source info
                primary_source = {
                    'file_name': file_name,
                    'page_no': page_no
                } if file_name else None
                
                return SearchResult(
                    instruction_index=instruction_index,
                    success=True,
                    content=context,
                    metadata={
                        'search_type': f'document_{search_mode}',
                        'query': query,
                        'all_sources': source_info  # Store all sources found
                    },
                    source_info=primary_source
                )
            else:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=False,
                    content="No relevant documents found",
                    metadata={'search_type': f'document_{search_mode}'},
                    error_message="No matching documents"
                )
                
        except Exception as e:
            raise Exception(f"Document search failed: {e}")
    
    def _execute_sql_search(self, query: str, user_session: str, 
                          instruction_index: int) -> SearchResult:
        """Execute SQL search on structured data."""
        try:
            sql_context = self.context_service.get_data_context(user_session, query)
            
            if sql_context:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=True,
                    content=sql_context,
                    metadata={
                        'search_type': 'sql_query',
                        'query': query
                    }
                )
            else:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=False,
                    content="No relevant data found",
                    metadata={'search_type': 'sql_query'},
                    error_message="No matching data"
                )
                
        except Exception as e:
            raise Exception(f"SQL search failed: {e}")
    
    def _execute_summary_search(self, query: str, user_session: str, 
                              instruction_index: int) -> SearchResult:
        """Execute summary-based search."""
        try:
            summary = self.context_service.get_summary_context(user_session, query)
            
            if summary:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=True,
                    content=summary,
                    metadata={
                        'search_type': 'summary',
                        'query': query
                    }
                )
            else:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=False,
                    content="No summary available",
                    metadata={'search_type': 'summary'},
                    error_message="No summary found"
                )
                
        except Exception as e:
            raise Exception(f"Summary search failed: {e}")
    
    def _execute_hybrid_search(self, query: str, user_session: str, 
                             instruction_index: int) -> SearchResult:
        """Execute hybrid search combining document and SQL contexts."""
        try:
            # Get both document and SQL contexts
            doc_context, file_name, page_no = self.context_service.get_document_context(
                user_session, query
            )
            sql_context = self.context_service.get_data_context(user_session, query)
            
            # Combine contexts
            combined_content = ""
            if doc_context:
                combined_content += f"Document Context:\n{doc_context}\n\n"
            if sql_context:
                combined_content += f"Data Context:\n{sql_context}\n\n"
            
            if combined_content:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=True,
                    content=combined_content,
                    metadata={
                        'search_type': 'hybrid',
                        'query': query,
                        'has_documents': bool(doc_context),
                        'has_data': bool(sql_context)
                    },
                    source_info={
                        'file_name': file_name,
                        'page_no': page_no
                    }
                )
            else:
                return SearchResult(
                    instruction_index=instruction_index,
                    success=False,
                    content="No relevant information found",
                    metadata={'search_type': 'hybrid'},
                    error_message="No matching content"
                )
                
        except Exception as e:
            raise Exception(f"Hybrid search failed: {e}")
    
    def _enhance_query_with_context(self, query: str, 
                                  context_accumulator: Dict[str, Any],
                                  instructions: str) -> str:
        """Enhance the search query with accumulated context."""
        if not context_accumulator:
            return query
        
        # Add context hints to the query if there are relevant previous results
        context_hints = []
        
        # Look for relevant previous findings
        for key, value in context_accumulator.items():
            if isinstance(value, str) and len(value) > 0:
                # Add a brief context hint
                context_hints.append(f"Previous finding from {key}: {value[:100]}...")
        
        if context_hints and len(context_hints) <= 2:  # Don't overwhelm the query
            enhanced_query = f"{query}\n\nContext from previous searches:\n" + "\n".join(context_hints[:2])
            return enhanced_query
        
        return query
    
    def _update_context_accumulator(self, results: List[SearchResult], 
                                   context_accumulator: Dict[str, Any]) -> None:
        """Update the context accumulator with new results."""
        for result in results:
            if result.success and result.content:
                search_type = result.metadata.get('search_type', 'unknown')
                key = f"{search_type}_{result.instruction_index}"
                
                # Store a summary of the result for context
                if len(result.content) > 500:
                    context_accumulator[key] = result.content[:500] + "..."
                else:
                    context_accumulator[key] = result.content
    
    def _estimate_execution_time(self, instructions: List[SearchInstruction]) -> float:
        """Estimate total execution time for the instructions."""
        # Simple estimation based on search types
        time_estimates = {
            SearchType.DOCUMENT_VECTOR: 2.0,
            SearchType.DOCUMENT_KEYWORD: 1.5,
            SearchType.SQL_QUERY: 3.0,
            SearchType.SUMMARY: 1.0,
            SearchType.HYBRID: 4.0
        }
        
        total_time = sum(
            time_estimates.get(instruction.search_type, 2.0) 
            for instruction in instructions
        )
        
        return total_time
    
    def deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate content from search results based on content similarity.
        
        Args:
            results: List of search results that may contain duplicates
            
        Returns:
            List of deduplicated search results
        """
        seen_content = {}
        deduplicated = []
        
        for result in results:
            if not result.success or not result.content.strip():
                continue
                
            # Create a content fingerprint by normalizing the text
            content_fingerprint = self._create_content_fingerprint(result.content)
            
            # Check if we've seen similar content
            if content_fingerprint not in seen_content:
                seen_content[content_fingerprint] = result
                deduplicated.append(result)
            else:
                # If duplicate, merge metadata if needed
                existing = seen_content[content_fingerprint]
                logger.info(f"Found duplicate content from {result.metadata.get('search_type')}"
                        f"(already seen in {existing.metadata.get('search_type')})")
        
        logger.info(f"Deduplication: {len(results)} results -> {len(deduplicated)} unique")
        return deduplicated

    def _create_content_fingerprint(self, content: str, threshold: int = 100) -> str:
        """
        Create a fingerprint for content to detect near-duplicates.
        
        Args:
            content: The content to fingerprint
            threshold: Number of characters to use for fingerprint
            
        Returns:
            A normalized fingerprint string
        """
        # Normalize whitespace and case
        normalized = ' '.join(content.lower().split())
        
        # Use first N characters as fingerprint (simple approach)
        if len(normalized) <= threshold:
            return normalized
        
        # Take beginning and end to catch similar chunks
        return normalized[:threshold//2] + "..." + normalized[-threshold//2:]

    def filter_and_rank_results(self, results: List[SearchResult], 
                            original_query: str) -> List[SearchResult]:
        """Filter, deduplicate, and rank results by relevance and success."""
        # Filter successful results
        successful_results = [r for r in results if r.success and r.content.strip()]
        
        # Deduplicate results
        deduplicated_results = self.deduplicate_results(successful_results)
        
        # Sort by priority and content length (as a proxy for relevance)
        def result_score(result: SearchResult) -> float:
            base_score = len(result.content) / 1000  # Content length score
            
            # Boost based on search type relevance
            type_boost = {
                'document_vector': 1.0,
                'hybrid': 0.9,
                'document_keyword': 0.8,
                'sql_query': 0.7,
                'summary': 0.6
            }
            
            search_type = result.metadata.get('search_type', 'unknown')
            boost = type_boost.get(search_type, 0.5)
            
            return base_score * boost
        
        deduplicated_results.sort(key=result_score, reverse=True)
        return deduplicated_results

# Create a singleton instance
_search_execution_service = None

def get_search_execution_service() -> SearchExecutionService:
    """
    Get the search execution service singleton instance.
    
    Returns:
        SearchExecutionService: The search execution service instance
    """
    global _search_execution_service
    if _search_execution_service is None:
        _search_execution_service = SearchExecutionService()
    return _search_execution_service