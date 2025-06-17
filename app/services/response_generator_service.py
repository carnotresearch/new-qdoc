"""
Response generator service.

This module provides response generation for different types of queries:
- General chat responses (no document context)
- Summary responses (using abstractive summaries)
- Document query responses (using document context)
- Data query responses (using SQL results)
- Hybrid responses (using multiple contexts)
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from app.services.llm_service import get_standard_llm, get_fast_llm

# Configure logging
logger = logging.getLogger(__name__)

class ResponseGeneratorService:
    """Service for generating responses to user queries."""
    
    def __init__(self):
        """Initialize the response generator service."""
        logger.info("Initializing response generator service")
        # Use standard LLM for most response generation tasks
        self.llm = get_standard_llm()
        # Use fast LLM for simple responses
        self.fast_llm = get_fast_llm()
        
    def generate_general_chat_response(self, user_query: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for general chat queries.
        
        Args:
            user_query: User's query
            language: Preferred response language
            
        Returns:
            Response data
        """
        # Create prompt for LLM
        prompt = f"""You are a chat bot called icarKno created by Carnot Research Pvt Ltd, which answers queries related to a document.
        The user is interacting with you but has not yet uploaded any documents or selected a knowledge container.
        Please respond naturally to the following question in a conversational tone. If appropriate, gently remind the user 
        they can upload files in the sidebar menu or select a knowledge container to ask document-specific questions.

        Question:
        ```{user_query}```
        """
        
        # Add language in prompt if not English
        if language and language != 'English':
            prompt = prompt + f"\nAnswer in user's preferred language - {language}."

        # Generate response from LLM - use fast LLM for simple responses
        try:
            logger.info(f'Generating general chat response with approx token count: {len(prompt.split()) * 1.33}')
            llm_response = self.fast_llm.invoke(prompt)
            logger.info(f'Generated general chat response')
            response = str(llm_response.content)
            
            return {"answer": response}
        except Exception as e:
            logger.error(f'Error generating general chat response: {e}')
            return {"answer": "I'm sorry, I encountered an error while processing your request. Please try again."}
    
    def generate_summary_response(self, summary_text: str) -> Dict[str, Any]:
        """
        Generate a response for summary queries.
        
        Args:
            summary_text: Summary text
            
        Returns:
            Response data
        """
        return {
            "answer": summary_text,
            "sources": [],
            "questions": []
        }
        
    def generate_document_response(self, user_query: str, context: str, 
                                 file_name: Optional[str] = None, 
                                 page_no: Optional[int] = None,
                                 language: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for document queries.
        
        Args:
            user_query: User's query
            context: Document context
            file_name: File name for the primary document
            page_no: Page number for the primary document
            language: Preferred response language
            
        Returns:
            Response data
        """
        # Create prompt for LLM
        prompt = self._create_document_prompt(user_query, context, language)
        
        # Generate response from LLM
        try:
            logger.info(f'Generating document response with approx token count: {len(prompt.split()) * 1.33}')
            llm_response = self.llm.invoke(prompt)
            logger.info(f'Generated document response: \n {llm_response.content}')
            response_text = str(llm_response.content)
            
            # Extract components from the response
            cleaned_response, sources, questions = self._extract_structured_response(response_text)
            
            # If no sources were extracted but we have file info, add it
            if not sources and file_name:
                sources = [{"fileName": file_name, "pageNo": page_no or 0}]
                
            return {
                "answer": cleaned_response, 
                "fileName": file_name or "", 
                "pageNo": page_no or 0, 
                "sources": sources, 
                "questions": questions
            }
        except Exception as e:
            logger.error(f'Error generating document response: {e}')
            return {
                "answer": "I encountered an error while processing your query. Please try again or rephrase your question.",
                "fileName": file_name or "",
                "pageNo": page_no or 0,
                "sources": [],
                "questions": []
            }
            
    def generate_data_response(self, user_query: str, sql_context: str, 
                            language: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for data queries.
        
        Args:
            user_query: User's query
            sql_context: SQL query results context
            language: Preferred response language
            
        Returns:
            Response data
        """
        # Create prompt for LLM
        prompt = f"""
        You are a JSON answer generator responding to questions about data from SQL tables. Follow these rules:

        1. Return response ONLY as valid JSON with this structure:
        {{
        "answer": "Full answer text explaining the data findings",
        "sources": [],
        "relevant_questions": ["question1?", ...]
        }}

        2. Response Guidelines:
        - Answer should clearly explain what was found in the data
        - Include any significant numbers, trends, or patterns
        - Do not mention anything about SQL or provided SQL query in the answer and questions, it is only for your reference
        - Indicate if the data was limited or if more data exists
        - Generate follow-up questions (max 3) which would help explore the data further

        SQL Context:
        {sql_context}

        User Question: {user_query}

        Generate ONLY the JSON response. Do not include any other text or explanations.
        """
        
        # Add language preference if provided
        if language and language != 'English':
            prompt += f"\nAnswer in user's preferred language - {language}."
        
        # Generate response from LLM
        try:
            logger.info(f'Generating data response with approx token count: {len(prompt.split()) * 1.33}')
            llm_response = self.llm.invoke(prompt)
            logger.info(f'Generated data response')
            response_text = str(llm_response.content)
            
            # Extract components from the response
            cleaned_response, sources, questions = self._extract_structured_response(response_text)
            
            return {
                "answer": cleaned_response, 
                "fileName": "", 
                "pageNo": 0, 
                "sources": sources, 
                "questions": questions
            }
        except Exception as e:
            logger.error(f'Error generating data response: {e}')
            return {
                "answer": "I encountered an error while analyzing the data. Please try again or rephrase your question.",
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
            
    def generate_hybrid_response(self, user_query: str, document_context: str, 
                               sql_context: str, file_name: Optional[str] = None, 
                               page_no: Optional[int] = None,
                               language: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for hybrid queries needing both document and data contexts.
        
        Args:
            user_query: User's query
            document_context: Document context
            sql_context: SQL query results context
            file_name: File name for the primary document
            page_no: Page number for the primary document
            language: Preferred response language
            
        Returns:
            Response data
        """
        # Create prompt for LLM
        prompt = f"""
        You are a JSON answer generator responding to questions that need both document and data analysis. Follow these rules:

        1. Return response ONLY as valid JSON with this structure:
        {{
        "answer": "Full answer text integrating document and data insights",
        "sources": [
            {{"fileName": "filename.pdf", "pageNo": number}},
            ... 
        ],
        "relevant_questions": ["question1?", ...]
        }}

        2. Response Guidelines:
        - Integrate insights from both the documents and the data tables
        - Cross-reference information when possible to provide a comprehensive answer
        - Explain any discrepancies or relationships between text and data
        - Do not mention about SQL query and tables in the answer and questions
        - Include ALL source citations used to answer the question
        - Generate follow-up questions (max 3) which can be answered based on the context
        - Use EXACT filename from source context

        Document Context:
        {document_context}

        Data Context:
        {sql_context}

        User Question: {user_query}

        Generate ONLY the JSON response. Do not include any other text or explanations.
        """
        
        # Add language preference if provided
        if language and language != 'English':
            prompt += f"\nAnswer in user's preferred language - {language}."
        
        # Generate response from LLM
        try:
            logger.info(f'Generating hybrid response with approx token count: {len(prompt.split()) * 1.33}')
            llm_response = self.llm.invoke(prompt)
            logger.info(f'Generated hybrid response')
            response_text = str(llm_response.content)
            
            # Extract components from the response
            cleaned_response, sources, questions = self._extract_structured_response(response_text)
            
            # If no sources were extracted but we have file info, add it
            if not sources and file_name:
                sources = [{"fileName": file_name, "pageNo": page_no or 0}]
                
            return {
                "answer": cleaned_response, 
                "fileName": file_name or "", 
                "pageNo": page_no or 0, 
                "sources": sources, 
                "questions": questions
            }
        except Exception as e:
            logger.error(f'Error generating hybrid response: {e}')
            return {
                "answer": "I encountered an error while processing your query. Please try again or rephrase your question.",
                "fileName": file_name or "",
                "pageNo": page_no or 0,
                "sources": [],
                "questions": []
            }
            
    def save_query_history(self, user_session: str, question: str, response: str) -> None:
        """
        Save the user question and LLM response for future context.
        
        Args:
            user_session: User's session identifier
            question: User's question
            response: LLM response
        """
        try:
            prevquestion_filename = f'users/{user_session}/prev_question.txt'
            os.makedirs(os.path.dirname(prevquestion_filename), exist_ok=True)
            with open(prevquestion_filename, 'w', encoding='utf-8') as file:
                file.write(f"Previous question in chat history: {question}\nPrevious Response in chat history: {response}")
            logger.debug(f"Saved query history for session {user_session}")
        except Exception as e:
            logger.error(f'Error saving query history: {e}')
    
    def _create_document_prompt(self, user_question: str, context: str, language: Optional[str] = None) -> str:
        """
        Generate a prompt for the LLM with inline citation requirements.
        
        Args:
            user_question (str): User's question
            context (str): Context for the question
            language (str, optional): Language for response
            
        Returns:
            str: Generated prompt
        """
        prompt = f"""
        You are a document analysis assistant that provides accurate, well-cited responses based on provided document excerpts.

        # RESPONSE FORMAT
        Return response ONLY as valid JSON with this structure:
        {{
        "answer": "Full answer text with inline citations",
        "sources": [
            {{"fileName": "filename.pdf", "pageNo": number}},
            ... 
        ],
        "relevant_questions": ["question1?", ...]
        }}

        # CITATION INSTRUCTIONS
        **CRITICAL**: You must cite sources inline throughout your answer. For every statement, fact, or piece of information you include:

        1. **Immediate Citation**: Add a citation immediately after each sentence or logical sequence that comes from the document context
        2. **Citation Format**: Use this exact format: [filename.pdf, Page X] 
        3. **Accuracy**: Only cite sources that are explicitly provided in the context below
        4. **No Hallucination**: Do not make up sources, page numbers, or filenames that are not in the provided context
        5. **Complete Coverage**: Every factual statement in your answer should have a corresponding citation

        # EXAMPLE OF PROPER CITATION:
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers [AI_Handbook.pdf, Page 15]. These networks can automatically learn hierarchical representations of data [ML_Fundamentals.docx, Page 23]. The technology has revolutionized computer vision and natural language processing [AI_Handbook.pdf, Page 16]."

        # RESPONSE GUIDELINES:
        - **For out-of-context questions**: 
        {{
            "answer": "That's out of context. Please try questions like these.",
            "sources": [],
            "relevant_questions": ["context-related question1?", ...]
        }}

        - **Answer Construction**:
        * Write clear, coherent sentences
        * Add inline citations immediately after relevant information
        * Ensure every citation corresponds to actual content in the context
        * Maintain natural flow while including precise source attribution

        - **Sources Array**: Include ALL unique sources used in your answer in the sources array
        - **Follow-up Questions**: Generate relevant questions (max 3) that can be answered from the context
        - **Verification**: Double-check that every citation in your answer corresponds to actual sources in the context

        # CONTEXT:
        {context}

        # USER QUESTION: 
        {user_question}

        Generate ONLY the JSON response. Do not include any other text or explanations.
        """
        
        # Add language preference if provided
        if language and language != 'English':
            prompt += f"\nAnswer in user's preferred language - {language}."
        
        return prompt
        
    def _extract_structured_response(self, response):
        """
        Extract answer, sources, and questions from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            tuple: (answer, sources, questions)
        """
        # Pre-clean response
        cleaned = re.sub(r'(?i)json\s*:', '', response)  # Remove JSON: prefixes
        cleaned = cleaned.strip("` \n")  # Remove code block markers
        
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[cleaned.index('{'):]

        # Attempt 1: Strict JSON parsing
        try:
            parsed = json.loads(cleaned)
            return self._validate_structure(parsed)
        except json.JSONDecodeError:
            pass
            
        # Attempt 2: Fix common syntax errors
        try:
            # Add quotes around unquoted keys
            repaired = re.sub(r'(?<!\\)(\w+)(\s*:)', r'"\1"\2', cleaned)
            # Fix trailing commas
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            parsed = json.loads(repaired)
            return self._validate_structure(parsed)
        except json.JSONDecodeError:
            pass

        # Attempt 3: Fallback to text parsing
        return self._extract_data(cleaned)

    def _validate_structure(self, parsed):
        """
        Ensure required fields exist with proper types.
        
        Args:
            parsed: Parsed JSON structure
            
        Returns:
            tuple: (answer, sources, questions)
        """
        return (
            parsed.get('answer', ''),
            parsed.get('sources', []),
            parsed.get('relevant_questions', [])
        )

    def _extract_data(self, response):
        """
        Extract answer, sources, and questions from structured text.
        
        Args:
            response: LLM response text
            
        Returns:
            tuple: (answer, sources, questions)
        """
        # Initialize defaults
        result = {
            'answer': response,  # Default to full response if no sections found
            'sources': [],
            'questions': []
        }
        
        try:
            # Define flexible section headers with regex patterns
            section_patterns = {
                'sources': re.compile(r'^\s*\*\*Sources:\*\*', re.IGNORECASE | re.MULTILINE),
                'questions': re.compile(r'^\s*\*\*(?:Relevant|Leading) Questions:\*\*', re.IGNORECASE | re.MULTILINE)
            }
            
            # Find all section positions
            sections = []
            for name, pattern in section_patterns.items():
                match = pattern.search(response)
                if match:
                    sections.append((name, match.start()))
            
            # Sort sections by their position in the text
            sections.sort(key=lambda x: x[1])
            
            if not sections:
                # No sections found, return full response as answer
                return result['answer'], [], []
            
            # Extract answer text (everything before first section)
            answer_start = 0
            if response[:sections[0][1]].strip().startswith('**Answer:**'):
                answer_start = response.find('**Answer:**') + len('**Answer:**')
            
            result['answer'] = response[answer_start:sections[0][1]].strip()
            
            # Process each section
            for i, (name, pos) in enumerate(sections):
                # Get section content (from current pos to next section start or end)
                end_pos = sections[i+1][1] if i+1 < len(sections) else len(response)
                section_content = response[pos:end_pos].strip()
                
                # Remove section header
                header_match = section_patterns[name].search(section_content)
                if header_match:
                    content = section_content[header_match.end():].strip()
                else:
                    content = section_content
                
                # Process based on section type
                if name == 'sources':
                    result['sources'] = self._extract_sources(content)
                elif name == 'questions':
                    result['questions'] = self._extract_questions(content)
            
        except Exception as e:
            logger.error(f"Error extracting structured response: {e}")
        
        logger.info(f'Extracted fields - Answer: {result["answer"]}, Sources: {result["sources"]}, Questions: {result["questions"]}')
        return result['answer'], result['sources'], result['questions']

    def _extract_sources(self, content):
        """
        Extract sources from formatted content.
        
        Args:
            content: Content containing source information
            
        Returns:
            list: List of source dictionaries
        """
        sources = []
        patterns = [
            # Matches "Source: file.pdf, Page 3" variants
            r'(?i)(?:source|sources|ref)[:\-*]\s*([^,]+?)(?:,\s*(?:pg?|page)[^\d]*(\d+))',
            # Matches bullet points
            r'[\-\*•]\s*([^,]+?)\s*[,\-]\s*(?:pg?|page)\s*(\d+)'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                filename = match.group(1).strip()
                page = match.group(2)
                # Clean filename
                filename = re.sub(r'^[\"\'\(]*|[\)\'\"]*$', '', filename)
                # Handle page numbers
                try:
                    page_no = int(''.join(filter(str.isdigit, page)))
                except ValueError:
                    page_no = 0
                sources.append({'fileName': filename, 'pageNo': page_no})
        
        return sources

    def _extract_questions(self, content):
        """
        Extract questions from formatted content.
        
        Args:
            content: Content containing question information
            
        Returns:
            list: List of question strings
        """
        questions = []
        # Split potential question blocks
        blocks = re.split(r'\n\s*\d+[\.\)]?|\n\s*[\-\*•]', content)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Remove quotation marks
            block = re.sub(r'^[\'"]|[\'"]$', '', block)
            # Validate question structure
            if re.search(r'\?$|^(how|what|when|where|why|who|can|does|do|is|are)', block, re.I):
                questions.append(block)
        
        return questions[:3]  # Return max 3 questions
    
    def generate_document_aware_chat_response(self, user_query: str, documents_info: Dict[str, Any],
                                       language: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response for general chat when documents are available.
        This handles cases where the user has selected documents but is asking general questions.
        
        Args:
            user_query: User's query
            documents_info: Information about available documents
            language: Preferred response language
            
        Returns:
            Response data
        """
        # Create prompt for LLM
        prompt = f"""You are a chat bot called icarKno created by Carnot Research Pvt Ltd.
        The user has uploaded or selected documents, and you're having a general conversation.
        
        Here's information about their documents:
        - Number of files: {documents_info.get('file_count', 'unknown')}
        - File types: {', '.join(documents_info.get('file_types', ['unknown']))}
        - Topics: {documents_info.get('topics', 'various')}
        
        The user is asking a general question that doesn't specifically require document context.
        Please respond naturally and conversationally, but you can briefly reference their documents 
        if relevant to the conversation. Don't force document references if they're not relevant.
        
        Question:
        ```{user_query}```
        """
        
        # Add language in prompt if not English
        if language and language != 'English':
            prompt = prompt + f"\nAnswer in user's preferred language - {language}."

        # Generate response from LLM - use fast LLM for simple responses
        try:
            llm_response = self.fast_llm.invoke(prompt)
            logger.info(f'Generated document-aware chat response')
            response = str(llm_response.content)
            
            return {
                "answer": response,
                "fileName": "",
                "pageNo": 0,
                "sources": [],
                "questions": []
            }
        except Exception as e:
            logger.error(f'Error generating document-aware chat response: {e}')
            return {"answer": "I'm sorry, I encountered an error while processing your request. Please try again."}

# Create a singleton instance
_response_generator_service = None

def get_response_generator_service() -> ResponseGeneratorService:
    """
    Get the response generator service singleton instance.
    
    Returns:
        ResponseGeneratorService: The response generator service instance
    """
    global _response_generator_service
    if _response_generator_service is None:
        _response_generator_service = ResponseGeneratorService()
    return _response_generator_service