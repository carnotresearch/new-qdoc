"""
Document querying and answering module.

This module provides functions for:
- Searching documents using vector and keyword search
- Generating responses to user queries using LLMs
- Extracting and formatting information from documents
"""

# Standard library imports
import logging
import os

# Third-party imports
from langchain_openai import ChatOpenAI

# Local imports
from controllers.sql_db import query_database
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
openai_api_key = Config.OPENAI_API_KEY


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