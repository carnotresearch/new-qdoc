import shelve
from dotenv import load_dotenv
import time
import logging
import yaml
import requests
from .sessionInfo import fetch_session_ids

# Import the LLM service
from utils.llm_service import get_openai_chat

load_dotenv()
session_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IndoYXRzYXBwQGNhcm5vdHJlc2VhcmNoLmNvbSIsImlhdCI6MTczNDAwNjMwOH0.0Qt6iW3ptchU22ZDmbDQvnREKZll9xo_0BySRKCm0_0"

# Create an OpenAI client from our LLM service
def get_client():
    # This creates a wrapped OpenAI client through our service
    # This ensures we're using the same API key and configuration
    return get_openai_chat().client

# Use context manager to ensure the shelf file is closed properly
def check_if_thread_exists(wa_id):
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)

def store_thread(wa_id, thread_id):
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = thread_id

def hit_icarkno(data):
    """Send POST request using configuration loaded from YAML."""
    # Load configuration from YAML
    url = "https://qdocbackend.carnotresearch.com/ask"
    headers = {
        "Content-Type": "application/json"
    }

    icarkno_data = {
        "token": session_token,
        "inputLanguage": "23",
        "outputLanguage": "23",
        "context": "files",
        "temperature": 0.7,
        "mode": "contextual",
        "hasCsvOrXlsx": False,
        "message": data["message"],
        "sessionId": data["sessionId"]
    }

    logging.info(f"ICARKNO_Data: {icarkno_data}")
    
    # Send POST request
    response = requests.post(url, headers=headers, json=icarkno_data)
    
    if response.status_code == 200:
        logging.info("icarKno Response:", response.json())
    else:
        logging.info(f"icarKno req {response.status_code}: {response.text}")
    
    response_data = response.json()  # Extracts the JSON data as a dictionary
    return response_data.get('answer')  # Returns the 'answer' from the dictionary

selected_session_id = None
def generate_response_icarkno(message_body, wa_id, name):
    global selected_session_id
    # List of common greetings
    greetings = ["hi", "hello", "hey", "hola", "namaste", "greetings"]

    print(f'message and selected session id: {message_body} {selected_session_id}')

    #Normalize the message to lowercase and strip whitespace
    if message_body.strip().lower() in greetings:
        normalized_message = message_body.strip().lower()
    else:
        normalized_message = message_body.strip()
    
    #fetch session IDs
    token = session_token
    sessions = fetch_session_ids(token)
    session_names = [session['name'] for session in sessions]
    session_mapping = {session['name'].lower(): session['id'] for session in sessions}  # Map names to IDs
    logging.info(f"Sessions: {sessions}")
    logging.info(f"Session Names: {session_names}")
        

    # Check if the message is a greeting
    if normalized_message in greetings:
        logging.info(f"Greeting detected: {message_body}")
        
        if not sessions:
            session_list = "No active sessions available at the moment."
        else:
            session_list = "\n".join([f"{index}: {session['name']}" for index, session in enumerate(sessions)])
        
        selected_session_id = None
        return f"{message_body} {name}, kindly choose your session by typing the session name.\nHere is the list of available sessions for you to choose:\n{session_list}"
    elif selected_session_id is None:
        logging.info(f"data types: {type(session_mapping)} {type(normalized_message)}")
        selected_name = str(normalized_message).lower()
        selected_id = session_mapping.get(selected_name.lower())  # Ensure case-insensitive matching
        if selected_id:
            logging.info(f"The ID for '{selected_name}' is: {selected_id}")
            selected_session_id = selected_id
            logging.info(f"selected session id: {selected_session_id}")
            return f"You've selected '{selected_name}'. How can I assist you?"
        else:
            try:
                selected_id = session_mapping.get(session_names[int(selected_name)].lower())
                if selected_id:
                    logging.info(f"The ID for '{selected_name}' is: {selected_id}")
                    selected_session_id = selected_id
                    logging.info(f"selected session id: {selected_session_id}")
                    return f"You've selected '{selected_name}'. How can I assist you?"
            except (ValueError, IndexError):
                pass
                
            logging.info(f"no valid session name: {message_body}")
            
            if not sessions:
                session_list = "No active sessions available at the moment."
            else:
                session_list = "\n".join([f"{index}: {session['name']}" for index, session in enumerate(sessions)])
            
            return f"{message_body} {name}, choose valid session by typing the session name.\nHere is the list of available sessions for you to choose:\n{session_list}"
        
    # Check if there is already a thread_id for the wa_id
    thread_id = check_if_thread_exists(wa_id)

    # Get OpenAI client from our service
    client = get_client()

    # If a thread doesn't exist, create one and store it
    if thread_id is None:
        logging.info(f"Creating new thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.create()
        store_thread(wa_id, thread.id)
        thread_id = thread.id

    # Otherwise, retrieve the existing thread
    else:
        logging.info(f"Retrieving existing thread for {name} with wa_id {wa_id}")
        thread = client.beta.threads.retrieve(thread_id)

    # Add message to thread
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message_body,
    )

    #Dynamically generate the payload   
    dynamic_data = {
        'sessionId':selected_session_id,
        'message': message_body,
        'userName': name,  # Example of an extra field not in the YAML
        'threadId': thread_id  # Example of thread info sent dynamically
    }
    logging.info(f"DYNAMIC DATA:{dynamic_data}")
    
    new_message=hit_icarkno(dynamic_data)

    return new_message