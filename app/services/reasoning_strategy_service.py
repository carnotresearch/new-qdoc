"""
Reasoning strategy service for creative mode query processing.

This module implements a chain-of-thought approach for analyzing user queries,
devising search strategies, and coordinating multiple information retrieval steps.
"""

import logging
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field

from app.services.llm_service import get_comprehensive_llm

# Configure logging
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of searches that can be performed."""
    DOCUMENT_VECTOR = auto()  # Vector search in documents
    DOCUMENT_KEYWORD = auto()  # Keyword search in documents
    SQL_QUERY = auto()  # Query structured data
    HYBRID = auto()  # Combination of document and SQL
    SUMMARY = auto()  # Summary-based search

class SearchPriority(Enum):
    """Priority levels for search execution."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()

@dataclass
class SearchInstruction:
    """Individual search instruction with context and priority."""
    query: str
    search_type: SearchType
    priority: SearchPriority
    instructions: str
    expected_info: str
    dependencies: Optional[List[int]] = None  # Indices of searches this depends on

class SearchStrategy(BaseModel):
    """Pydantic model for search strategy validation."""
    reasoning: str = Field(description="Detailed chain of thought reasoning for the strategy")
    query_breakdown: List[str] = Field(
        description="List of 2-5 specific sub-questions that break down the original query",
        min_items=1,
        max_items=5
    )
    searches: List[Dict[str, Any]] = Field(
        description="List of search instructions with query, search_type, priority, instructions, and expected_info",
        min_items=1,
        max_items=5
    )
    synthesis_approach: str = Field(description="Detailed explanation of how to combine the search results")
    confidence_level: str = Field(description="Expected confidence level: high, medium, low")

class ReasoningStrategyService:
    """Service for creating and managing reasoning strategies for queries."""
    
    def __init__(self):
        """Initialize the reasoning strategy service."""
        logger.info("Initializing reasoning strategy service")
        # Use comprehensive LLM for complex strategy creation
        self.llm = get_comprehensive_llm()
        self.parser = PydanticOutputParser(pydantic_object=SearchStrategy)
    
    def analyze_query_and_create_strategy(self, user_query: str, 
                                        available_resources: Dict[str, bool],
                                        context_info: Optional[Dict[str, Any]] = None) -> SearchStrategy:
        """
        Analyze the user query and create a comprehensive search strategy.
        
        Args:
            user_query: The user's question
            available_resources: Dict indicating what resources are available
            context_info: Additional context about the user's session
            
        Returns:
            SearchStrategy object with detailed plan
        """
        logger.info(f"Creating reasoning strategy for query: {user_query}")
        
        # Create the strategy prompt
        prompt = self._create_strategy_prompt(user_query, available_resources, context_info)
        
        try:
            # Get strategy from LLM
            response = self.llm.invoke(prompt)
            logger.info(f"LLM Strategy Response: {response.content}")
            strategy = self.parser.parse(response.content)
            
            # Validate confidence level manually for compatibility
            if hasattr(strategy, 'confidence_level'):
                if strategy.confidence_level.lower() not in ['high', 'medium', 'low']:
                    strategy.confidence_level = 'medium'  # Default fallback
                else:
                    strategy.confidence_level = strategy.confidence_level.lower()
            
            logger.info(f"Created strategy with {len(strategy.searches)} search steps")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            # Fallback to simple strategy
            return self._create_fallback_strategy(user_query, available_resources)
    
    def _create_strategy_prompt(self, user_query: str, 
                              available_resources: Dict[str, bool],
                              context_info: Optional[Dict[str, Any]] = None) -> str:
        """Create a detailed prompt for strategy generation."""
        
        resource_description = self._describe_available_resources(available_resources)
        context_description = self._describe_context(context_info)
        
        prompt = f"""You are an expert research strategist tasked with creating a comprehensive multi-step search strategy to answer complex user questions. Your goal is to break down the query into logical sub-questions and create an optimal search plan.

AVAILABLE RESOURCES:
{resource_description}

CONTEXT INFORMATION:
{context_description}

SEARCH TYPES AVAILABLE:
- DOCUMENT_VECTOR: Semantic search through uploaded documents (best for finding conceptual information, explanations, and contextual content)
- DOCUMENT_KEYWORD: Keyword-based search through documents (best for finding specific terms, names, or exact phrases)
- HYBRID: Combination of document and data searches (best when answer requires both textual context and data insights)
- SUMMARY: Search through document summaries (best for getting overview or when full documents are too detailed)

QUERY DECOMPOSITION RULES:
1. **Analyze Complexity**: Break complex queries into 2-5 logical sub-questions
2. **Identify Information Types**: Determine what types of information are needed (facts, data, explanations, comparisons, etc.)
3. **Choose Optimal Search Types**: Select the most appropriate search type for each sub-question
4. **Plan Dependencies**: Some searches may need results from previous searches
5. **Prioritize Searches**: Order searches by importance and logical flow

SEARCH STRATEGY REQUIREMENTS:
- Maximum 5 search steps
- Each search should have a clear, specific purpose
- Consider dependencies between searches (e.g., Search 2 uses results from Search 1)
- Prioritize searches that are most likely to contain the core answer
- If the query needs both factual data and conceptual understanding, plan accordingly
- For comparative questions, plan multiple searches to gather different perspectives
- For analytical questions, plan searches that build upon each other

EXAMPLES OF GOOD DECOMPOSITION:

Example 1 - Complex Query: "How does our Q3 revenue compare to industry benchmarks and what factors contributed to any differences?"
Sub-questions:
1. "Quarter Q3 revenue" (DOCUMENT_KEYWORD - get company data)
2. "What are the industry Q3 revenue benchmarks?" (DOCUMENT_VECTOR - find industry reports)
3. "What factors affected our Q3 performance?" (HYBRID - combine financial data with business context)

Example 2 - Multi-faceted Query: "Explain the relationship between climate change and renewable energy adoption trends"
Sub-questions:
1. "Climate change impacts" (DOCUMENT_KEYWORD - get climate science info)
2. "What are the current renewable energy adoption trends?" (DOCUMENT_VECTOR - find recent reports)
3. "How do climate policies influence renewable energy adoption?" (DOCUMENT_VECTOR - policy documents)

USER QUESTION: {user_query}

ANALYSIS TASK:
1. **REASONING**: Provide step-by-step analysis of what information is needed
2. **QUERY_BREAKDOWN**: List 2-5 specific sub-questions that address different aspects
3. **SEARCH_PLAN**: For each sub-question, specify the search type and detailed instructions
4. **SYNTHESIS_APPROACH**: Explain how the results should be combined
5. **CONFIDENCE_LEVEL**: Assess expected success level

{self.parser.get_format_instructions()}

Provide your response as valid JSON only. Focus on creating a logical, efficient search strategy that maximizes the chance of finding comprehensive information."""

        return prompt
    
    def _describe_available_resources(self, resources: Dict[str, bool]) -> str:
        """Describe what resources are available for searching."""
        descriptions = []
        
        if resources.get('has_documents', False):
            descriptions.append("✓ Documents: PDF, DOCX, TXT files with full-text search capability")
        else:
            descriptions.append("✗ Documents: No documents available")
            
        if resources.get('has_data_tables', False):
            descriptions.append("✓ Data Tables: CSV/Excel files with SQL query capability")
        else:
            descriptions.append("✗ Data Tables: No structured data available")
            
        if resources.get('has_summaries', False):
            descriptions.append("✓ Summaries: Extractive summaries of documents available")
        else:
            descriptions.append("✗ Summaries: No summaries available")
        
        return "\n".join(descriptions)
    
    def _describe_context(self, context_info: Optional[Dict[str, Any]]) -> str:
        """Describe the context information."""
        if not context_info:
            return "No additional context available"
        
        descriptions = []
        
        if 'file_count' in context_info:
            descriptions.append(f"File count: {context_info['file_count']}")
        if 'file_types' in context_info:
            descriptions.append(f"File types: {', '.join(context_info['file_types'])}")
        if 'domain' in context_info:
            descriptions.append(f"Domain: {context_info['domain']}")
        
        return "\n".join(descriptions) if descriptions else "No specific context available"
    
    def _create_fallback_strategy(self, user_query: str, 
                                available_resources: Dict[str, bool]) -> SearchStrategy:
        """Create a simple fallback strategy when the main strategy creation fails."""
        logger.info("Creating fallback strategy")
        
        searches = []
        
        # If documents are available, start with document search
        if available_resources.get('has_documents', False):
            searches.append({
                "query": user_query,
                "search_type": "DOCUMENT_VECTOR",
                "priority": "HIGH",
                "instructions": "Find relevant information from documents",
                "expected_info": "Direct answer or relevant context"
            })
        
        # If data tables are available, add SQL search
        if available_resources.get('has_data_tables', False):
            searches.append({
                "query": user_query,
                "search_type": "SQL_QUERY", 
                "priority": "MEDIUM",
                "instructions": "Query structured data for numerical or factual information",
                "expected_info": "Data-driven insights"
            })
        
        return SearchStrategy(
            reasoning="Fallback strategy: Simple search approach due to strategy creation error",
            query_breakdown=[user_query],
            searches=searches,
            synthesis_approach="Combine available results with priority on document content",
            confidence_level="medium"
        )
    
    def convert_to_search_instructions(self, strategy: SearchStrategy) -> List[SearchInstruction]:
        """Convert strategy to executable search instructions."""
        instructions = []
        
        for i, search_dict in enumerate(strategy.searches):
            try:
                logger.info(f"Processing search instruction {i}: {search_dict}")
                logger.info(f"Raw search_type value: '{search_dict.get('search_type')}'")
                logger.info(f"Raw priority value: '{search_dict.get('priority')}'")

                search_type = SearchType[search_dict.get('search_type', 'DOCUMENT_VECTOR')]
                priority_raw = search_dict.get('priority', 'MEDIUM')
                priority_str = str(priority_raw).upper()
                if priority_str.isdigit():
                    priority_map = {'1': 'HIGH', '2': 'MEDIUM', '3': 'LOW'}
                    priority_str = priority_map.get(priority_str, 'MEDIUM')
                priority = SearchPriority[priority_str]
                
                instruction = SearchInstruction(
                    query=search_dict.get('query', ''),
                    search_type=search_type,
                    priority=priority,
                    instructions=search_dict.get('instructions', ''),
                    expected_info=search_dict.get('expected_info', ''),
                    dependencies=search_dict.get('dependencies')
                )
                instructions.append(instruction)
                
            except (KeyError, ValueError) as e:
                logger.warning(f"Error parsing search instruction {i}: {e}")
                continue
        
        return instructions
    
    def should_use_creative_mode(self, user_query: str, mode: str) -> bool:
        """Determine if creative mode should be used for this query."""
        return mode == 'creative'
    
    def estimate_complexity(self, user_query: str) -> str:
        """Estimate the complexity of the user query."""
        # Simple heuristics for complexity estimation
        word_count = len(user_query.split())
        
        # Check for complexity indicators
        complex_indicators = [
            'compare', 'analyze', 'relationship', 'correlation', 'trend',
            'multiple', 'various', 'different', 'both', 'either',
            'how does', 'what is the relationship', 'explain the difference'
        ]
        
        has_complex_indicators = any(indicator in user_query.lower() for indicator in complex_indicators)
        
        if word_count > 20 or has_complex_indicators:
            return "high"
        elif word_count > 10:
            return "medium"
        else:
            return "low"

# Create a singleton instance
_reasoning_strategy_service = None

def get_reasoning_strategy_service() -> ReasoningStrategyService:
    """
    Get the reasoning strategy service singleton instance.
    
    Returns:
        ReasoningStrategyService: The reasoning strategy service instance
    """
    global _reasoning_strategy_service
    if _reasoning_strategy_service is None:
        _reasoning_strategy_service = ReasoningStrategyService()
    return _reasoning_strategy_service