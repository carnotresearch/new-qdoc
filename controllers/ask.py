"""
Document querying and answering module.

This module provides functions for:
- Searching documents using vector and keyword search
- Generating responses to user queries using LLMs
- Extracting and formatting information from documents
"""

# Standard library imports
import json
import logging
import os
import re
import time

# Third-party imports
from elastic.retriever import ElasticRetriever
from langchain_openai import ChatOpenAI
from werkzeug.utils import secure_filename

# Local imports
from controllers.sql_db import query_database
from controllers.doc_summary import summarize_document
from utils.extractText import clean_filename
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
openai_api_key = Config.OPENAI_API_KEY

# ----------------------------------------
# Context Retrieval Functions
# ----------------------------------------

def checkPreviousQn(usersession):
    """
    Retrieve the previous question asked by the user.
    
    Args:
        usersession (str): User's session identifier
        
    Returns:
        str: The previous question or empty string if none
    """
    prevquestion_filename = f'users/{usersession}/prev_question.txt'
    
    if os.path.exists(prevquestion_filename):
        with open(prevquestion_filename, 'r', encoding='utf-8') as file:
            prevqn = str(file.read())
    else:
        prevqn = ""
    
    return prevqn


def get_relevant_context(usersession, user_question):
    """
    Get relevant context for user question using elastic retriever.

    Args:
        usersession (str): User's session identifier as index name
        user_question (str): User's question
    
    Returns:
        tuple: Formatted context, file name, and page number
    """
    try:
        # Retrieve documents using Elastic search
        retriever = ElasticRetriever(usersession)
        docs = retriever.search(user_question)

        if not docs:
            return None, None, None
        
        # Extract page number and file name of the first document
        pageNo = 0
        fileName = ''
        try:
            actual_metadata = docs[0].metadata.get('_source', {}).get('metadata', {})
            # Get filename directly from metadata or from source path
            fileName = actual_metadata.get('filename', '')
            if not fileName:
                source = actual_metadata.get('source', '')
                fileName = clean_filename(source)
            pageNo = actual_metadata.get('page', 0) + 1
            logger.info(f'File name: {fileName}')
            logger.info(f'Page number: {pageNo}')
        except Exception as e: 
            logger.error(f'Error extracting metadata: {e}')
        
        # Process document information
        try:
            docs = extract_doc_info(docs)[:10]
        except Exception as e:
            logger.error(f'Error restructuring documents: {e}')

        # Format context for LLM
        formatted_context = ""
        for i, chunk in enumerate(docs, start=1):
            try:
                formatted_context += f"[{i}] \"{chunk['text']}\"  \n(Source: {chunk['source']}, Page {chunk['page']})\n\n"
            except Exception as e:
                logger.error(f'Error formatting context: {e}')
                continue
        
        return formatted_context, fileName, pageNo
    except Exception as e:
        logger.error(f'Error retrieving context: {e}')
        return None, None, None


def extract_doc_info(docs):
    """
    Extract and process information from search results.
    
    Args:
        docs (list): List of document objects from search results
        
    Returns:
        list: Processed list of document information
    """
    extracted = []
    
    try:
        # Set threshold score as median of first few documents
        threshold_score = (
            docs[1].metadata.get("_score") or 
            docs[1].metadata.get("_source", {}).get("_score") or 
            3
        )
    except Exception as e:  
        logger.warning(f'Error determining threshold score: {e}')
        threshold_score = 3
    
    for doc in docs:
        try:
            # Extract metadata
            meta = (
                doc.metadata.get('_source', {}).get('metadata', {}) 
                if '_source' in doc.metadata else doc.metadata
            )
            source = meta.get('source', '')
            filename = meta.get('filename', '')  # Get filename directly from metadata
            page = meta.get('page', 0) + 1
            
            # If filename is missing but source exists, extract clean filename from source
            if not filename and source:
                filename = clean_filename(source)
            
            logger.debug(f'File name: {filename}, page number: {page}')
            
            # Add document info to extracted list
            extracted.append({
                "text": doc.page_content,
                "source": filename,
                "page": page
            })
        except Exception as e:
            logger.warning(f"Failed to extract from doc: {e}")
    
    return extracted


# ----------------------------------------
# Query Intent Classification
# ----------------------------------------

def is_summary_query(query):
    """
    Determine if a user query is requesting a document summary.
    
    Args:
        query (str): User's question
        
    Returns:
        bool: True if the query is requesting a summary, False otherwise
    """
    prompt = f"""User is asking questions regarding a document, which can be in any format. If the user question is requesting a general summary of the entire document, respond with the single integer 1. If the question asks to summarize a specific section, extract information, or answer specific queries about the document, respond with 0. For general conversation or if any confusion, respond with 0. Only respond with the single integer 0 or 1 as the answer.

    Examples:

    Question: Who is Bill Gates?
    Expected Response: 0

    Question: Summarize this PDF in 5 sentences.
    Expected Response: 1

    Question: Summarize the conclusion of this document.
    Expected Response: 0

    Question: Summarize this.
    Expected Response: 1

    Question: Can you extract the key points from page 10?
    Expected Response: 0

    Question: Hi, how are you?
    Expected Response: 0

    User question:
    {query}"""
    
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
        response = llm.invoke(prompt)
        logger.info(f'Summary classification response: {response.content}')
        
        num = int(response.content)
        return bool(num)
    except Exception as e:
        logger.error(f'Error classifying query intent: {e}')
        return False


# ----------------------------------------
# Response Processing Functions
# ----------------------------------------

def extract_structured_response(response):
    """
    Extract answer, sources, and questions from LLM response.
    
    Args:
        response (str): LLM response text
        
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
        return validate_structure(parsed)
    except json.JSONDecodeError:
        pass
        
    # Attempt 2: Fix common syntax errors
    try:
        # Add quotes around unquoted keys
        repaired = re.sub(r'(?<!\\)(\w+)(\s*:)', r'"\1"\2', cleaned)
        # Fix trailing commas
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        parsed = json.loads(repaired)
        return validate_structure(parsed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: Fallback to text parsing
    return extract_data(cleaned)

def validate_structure(parsed):
    """
    Ensure required fields exist with proper types.
    
    Args:
        parsed (dict): Parsed JSON structure
        
    Returns:
        tuple: (answer, sources, questions)
    """
    return (
        parsed.get('answer', ''),
        parsed.get('sources', []),
        parsed.get('relevant_questions', [])
    )

def extract_data(response):
    """
    Extract answer, sources, and questions from structured text.
    
    Args:
        response (str): LLM response text
        
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
                result['sources'] = _extract_sources(content)
            elif name == 'questions':
                result['questions'] = _extract_questions(content)
        
    except Exception as e:
        logger.error(f"Error extracting structured response: {e}")
    
    return result['answer'], result['sources'], result['questions']

def _extract_sources(content):
    """
    Extract sources from formatted content.
    
    Args:
        content (str): Content containing source information
        
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

def _extract_questions(content):
    """
    Extract questions from formatted content.
    
    Args:
        content (str): Content containing question information
        
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


# ----------------------------------------
# Main Query Processing Functions
# ----------------------------------------

def user_input(user_question, usersession, isRegularQuery, hascsvxl, mode, filenames=None, language=None):
    """
    Process user queries and generate responses.
    
    Args:
        user_question (str): User's question
        usersession (str): User's session identifier
        isRegularQuery (bool): Flag for regular queries
        hascsvxl (bool): Flag for CSV/Excel data
        mode (str): Processing mode
        filenames (list, optional): List of filenames to filter results by
        language (str, optional): Language for response
        
    Returns:
        dict: Response with answer, filename, page number, sources, and questions
    """
    start_time = time.time()
    logger.info(f'Processing query: "{user_question}" for session {usersession}')

    # Check if query is for document summarization
    try:
        is_summary = is_summary_query(user_question)
        if is_summary:
            logger.info(f'Query identified as summary request')
            # Convert filenames to folder names if filenames are provided
            folder_names = []
            if filenames and len(filenames) > 0:
                for filename in filenames:
                    # Convert filename to secure folder name format
                    # This should match the logic used when storing files in upload.py
                    safe_folder_name = secure_filename(os.path.splitext(filename)[0])
                    if safe_folder_name:
                        folder_names.append(safe_folder_name)
                        
            logger.info(f"Using folder names for summary: {folder_names}")
                        
            # Use the modified summarize_document function
            summarized_content = summarize_document(user_question, usersession, language, folder_names)
            return {"answer": summarized_content}
    except Exception as e:
        logger.error(f'Error generating summary: {e}')
        logger.info(f'Could not create summary, continuing with normal flow')

    # Get relevant context for user question
    pageNo = 0
    fileName = ''
    formatted_context = ""
    try:
        formatted_context, fileName, pageNo = get_relevant_context(usersession, user_question)
        if not formatted_context:
            formatted_context = ""
    except Exception as e:
        logger.error(f'Error getting relevant context: {e}')    
    logger.info('---- %s seconds to do similarity and keyword search ----' % (time.time() - start_time))
    
    # Add SQL results if CSV/Excel files exist
    sqldoc = None
    if hascsvxl == True:
        sqldoc, error = query_database(usersession, user_question)
        if sqldoc:
            logger.info(f'SQL query results added to context')
            formatted_context = sqldoc + "\n\n" + formatted_context
        else:
            logger.info(f'No results found for SQL query')

    # Get previous question for context
    prevqn = checkPreviousQn(usersession=usersession)
    
    # Create prompt for LLM
    prompt = create_llm_prompt(user_question, formatted_context, language)
    
    # Generate response from LLM
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
        response = llm.invoke(prompt)
        logger.info(f'Generated LLM response')
        response_text = str(response.content)
        
        # Extract components from the response
        cleaned_response, sources, questions = extract_structured_response(response_text)
        
        # Handle dictionary response type
        if isinstance(cleaned_response, dict):
            sources = cleaned_response.get("sources", [])
            questions = cleaned_response.get("questions", [])
            cleaned_response = cleaned_response.get("answer", "")
        
        # Special handling for summary queries
        if is_summary_query(user_question):
            cleaned_response = f"{cleaned_response}\n\nDetailed summary is being generated and will be available in a few minutes."
            return {"answer": cleaned_response}

        # Save the response for future context
        save_query_history(usersession, user_question, response_text)
        
        logger.info('--- %s seconds to complete query processing ---' % (time.time() - start_time))
        
        return {
            "answer": cleaned_response, 
            "fileName": fileName, 
            "pageNo": pageNo, 
            "sources": sources, 
            "questions": questions
        }
    except Exception as e:
        logger.error(f'Error generating LLM response: {e}')
        return {
            "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",
            "fileName": fileName,
            "pageNo": pageNo,
            "sources": [],
            "questions": []
        }


def create_llm_prompt(user_question, context, language=None):
    """
    Generate a prompt for the LLM.
    
    Args:
        user_question (str): User's question
        context (str): Context for the question
        language (str, optional): Language for response
        
    Returns:
        str: Generated prompt
    """
    prompt = f"""
        You are a JSON answer generator. Answer user question from given excerpts from documents. Follow these rules:

        1. Return response ONLY as valid JSON with this structure:
        {{
        "answer": "Full answer text",
        "sources": [
            {{"fileName": "filename.pdf", "pageNo": number}},
            ... 
        ],
        "relevant_questions": ["question1?", ...]
        }}

        2. Response Guidelines:
        - For out-of-context questions: 
        {{
            "answer": "That's out of context. Please try questions like these.",
            "sources": [],
            "relevant_questions": ["context-related question1?", ...]
        }}

        - Infer user intent from the question and provide a relevant answer.
        - Include ALL source citations used to answer the question.
        - Generate follow-up questions (max 3) which can be answered based on the context . Do not generate any out of context questions.
        - If there is no context available, do not generate any questions.
        - Use EXACT filename from source context.

        Context:
        {context}

        User Question: {user_question}

        Generate ONLY the JSON response. Do not include any other text or explanations.
        """
    
    # Add language preference if provided
    if language and language != 'English':
        prompt += f"\nAnswer in user's preferred language - {language}."
    
    return prompt


def save_query_history(usersession, question, response):
    """
    Save the user question and LLM response for future context.
    
    Args:
        usersession (str): User's session identifier
        question (str): User's question
        response (str): LLM response
    """
    try:
        prevquestion_filename = f'users/{usersession}/prev_question.txt'
        os.makedirs(os.path.dirname(prevquestion_filename), exist_ok=True)
        with open(prevquestion_filename, 'w', encoding='utf-8') as file:
            file.write(f"Previous question in chat history: {question}\nPrevious Response in chat history: {response}")
        logger.debug(f"Saved query history for session {usersession}")
    except Exception as e:
        logger.error(f'Error saving query history: {e}')


# ----------------------------------------
# Translation Functions (Placeholder)
# ----------------------------------------

def translate_text(content, input_language, output_language):
    """
    Translate text between languages (placeholder implementation).
    
    Args:
        content (str): Text to translate
        input_language (int): Source language code
        output_language (int): Target language code
        
    Returns:
        str: Translated text
    """
    # TODO: Implement translation using Bhashini API
    return content


def get_language(language_code):
    """
    Get the language name from its code.
    
    Args:
        language_code (int): Language code
        
    Returns:
        str: Language name
    """
    languages = {
        1: "Hindi",
        2: "Gom",
        3: "Kannada",
        4: "Dogri",
        5: "Bodo",
        6: "Urdu",
        7: "Tamil",
        8: "Kashmiri",
        9: "Assamese",
        10: "Bengali",
        11: "Marathi",
        12: "Sindhi",
        13: "Maithili",
        14: "Punjabi",
        15: "Malayalam",
        16: "Manipuri",
        17: "Telugu",
        18: "Sanskrit",
        19: "Nepali",
        20: "Santali",
        21: "Gujarati",
        22: "Odia",
        23: "English", 
    }
    return languages.get(language_code, 'English')  


# ----------------------------------------
# Public API Functions
# ----------------------------------------

def get_demo_response(user_query, session_name):
    """
    Generate responses for demo mode (public transport queries).
    
    Args:
        user_query (str): User's question
        session_name (str): User's session identifier
        
    Returns:
        str: Generated response
    """
    user_query = user_query.strip()
    if not user_query:
        return 'Please provide a valid question.'
    
    try:
        # Get response from database
        sqldoc, _ = query_database(session_name, user_query)

        # Get previous question for context
        prev_question = checkPreviousQn(session_name)
        
        # Create prompt for LLM
        prompt = f"""You are a smart and intelligent public service assistant, icarKno (I carry Knowledge), created by Carnot Research Pvt Ltd.
        Answer user queries politely. Assist users in public transport queries and journey planning. 
        If queries are related to metro transportation, refer to relevant sql output given below to get station and facilities related information.
        Do not mention anything about SQL in the response. The SQL output is only for your reference.
        Answer smartly and in short to help user in his preferred language.

        SQL Output:
        ```{sqldoc}```

        Previous question in chat history:
        ```{prev_question}```

        User question:
        ```{user_query}```
        """
        
        # Generate response from LLM
        llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
        llm_response = llm.invoke(prompt)
        logger.info(f'Generated demo response')
        response = str(llm_response.content)
        return response
    except Exception as e:
        logger.error(f'Error generating demo response: {e}')
        raise


def get_llm_response(user_query, input_language, output_language, session_name, hascsvxl, mode, filenames=None):
    """
    Process user queries and generate responses.
    
    Args:
        user_query (str): User's question
        input_language (int): Input language code
        output_language (int): Output language code
        session_name (str): User's session identifier
        hascsvxl (bool): Flag for CSV/Excel data
        mode (str): Processing mode
        filenames (list, optional): List of filenames to filter results by
        
    Returns:
        dict: Response with answer and metadata
    """
    # Get language name
    language = get_language(output_language)
    
    # Validate query
    if not user_query or not user_query.strip():
        return {"answer": "Please provide a question."}
    
    # Process query
    try:
        response = user_input(
            user_query, 
            session_name, 
            True, 
            hascsvxl=hascsvxl, 
            mode=mode, 
            filenames=filenames, 
            language=language
        )
        return response
    except Exception as e:
        logger.error(f'Error processing query: {e}')
        raise


def get_general_llm_response(user_query, input_language, output_language, session_name):
    """
    Generate responses when no documents are uploaded.
    
    Args:
        user_query (str): User's question
        input_language (int): Input language code
        output_language (int): Output language code
        session_name (str): User's session identifier
        
    Returns:
        dict: Response with answer
    """
    # Get language name
    language = get_language(output_language)
    
    # Validate query
    if not user_query or not user_query.strip():
        return {"answer": "Please provide a question."}
    
    try:
        # Create prompt for LLM
        prompt = f"""You are a chat bot called Iknow created by Carnot Research Pvt Ltd, which answers queries related to a document.
        Right now, user has not provided the document and simply interacting with chat bot.
        So, just answer the below user question in short and ask the user to upload files in the sidebar menu on left or select a knowledge container from the existing container and tell the user than either no files are uploaded or no knowledge container is selected with existing files.
        Guide the user to upload files or select a knowledge container by telling them to select a knowledge container from the left menu or upload a file with the new container button.

        Question:
        ```{user_query}```
        """
        
        # Add language in prompt if language is not English
        if language and language != 'English':
            prompt = prompt + f"Answer in user's preferred language - {language}."

        # Generate response from LLM
        llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
        llm_response = llm.invoke(prompt)
        logger.info(f'Generated general response for empty document state')

        response = str(llm_response.content)
        
        # Translate output if needed (currently a placeholder)
        if output_language != 23:
            response = translate_text(response, 23, output_language)

        return {"answer": response}
    except Exception as e:
        logger.error(f'Error generating general response: {e}')
        raise