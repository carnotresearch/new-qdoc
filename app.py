"""
Flask backend application for document processing, analysis and querying.

This application provides REST API endpoints for:
- File upload and processing (PDF, DOCX, TXT, CSV, XLSX)
- Document querying using vector search and LLMs
- User authentication and management
- WhatsApp webhook integration

The application uses Elasticsearch for vector search and OpenAI for LLM queries.
"""

# Standard library imports
import asyncio
import json
import logging
import threading
import time

# Third-party imports
from elasticsearch import Elasticsearch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from sqlalchemy import create_engine

# Local application imports
from controllers.ask import get_llm_response, get_general_llm_response, get_demo_response
from controllers.database import (
    is_user_limit_over, create_fingerprint_entry, check_trial_status,
    upgrade_account, get_account_status, is_trial_limit_over
)
from controllers.sql_db import create_database_with_tables, store_table_info, add_tables_to_existing_db
from controllers.doc_summary import create_abstractive_summary
from controllers.upload import store_vector
from controllers.delete_session import delete_session
from utils.extractText import get_text_from_files
from utils.guardrails import input_guardrail_pipeline
from webhook import webhook_blueprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = 'supersecretkey'

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
)

def get_or_create_eventloop():
    """
    Get the current asyncio event loop or create a new one if none exists.
    
    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Initialize asyncio event loop
get_or_create_eventloop()

def authenticate(token):
    """
    Authenticate a user with the provided JWT token.
    
    Args:
        token (str): JWT token to authenticate
        
    Returns:
        str: User email extracted from the token
        
    Raises:
        jwt.ExpiredSignatureError: If the token has expired
        jwt.InvalidTokenError: If the token is invalid
    """
    data = jwt.decode(token, 'secret', algorithms=["HS256"])
    current_user = data['email']
    logging.info(f'User email: {current_user}')
    return current_user

@app.route('/freeTrial', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def freeTrial(): 
    """
    Handle file uploads for users in free trial mode.
    
    This endpoint processes each file individually:
    1. Validates the user's fingerprint
    2. Processes each uploaded file separately
    3. Stores the extracted text as vector embeddings
    4. Creates file-specific summaries asynchronously
    
    Returns:
        JSON response with status, message, and file details
    """
    start_time = time.time()
    
    try:
        # Extract and validate fingerprint
        fingerprint = request.form.get('fingerprint')
        create_fingerprint_entry(fingerprint)
        files = request.files.getlist("files")
        logging.info(f'Fingerprint and file name: {fingerprint}, {files[0].filename}')
    except Exception as e:
        logging.info(f'Error validating fingerprint: {e}')
        return jsonify({'message': 'Error validating fingerprint'}), 400
    
    # Check if any files were uploaded
    if not files or files[0].filename == '':
        message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
        return jsonify({'message': message}), 400
    
    # Separate files into valid document types and CSV files
    valid_files = [f for f in files if f.filename.endswith(('.pdf', '.docx', '.txt'))]
    csv_files = [f for f in files if f.filename.endswith(('.xlsx', '.csv'))]
    
    # Process CSV files if any
    if csv_files:
        # Add CSV files to the database
        return jsonify({'message': 'CSV files are not supported in free trial mode.'}), 400
    
    # delete previous session if exists
    try:
        delete_session(fingerprint)
    except Exception as e:
        logging.info(f'Error deleting previous session: {e}')
        
    # Process valid document files
    file_results = []
    message = ''
    if valid_files:
        try:
            # Process each file individually
            all_docs = []
            file_infos = []
            
            for file in valid_files:
                # Extract text from the file
                docs, file_info = get_text_from_files([file])
                if docs:
                    all_docs.append(docs)
                    file_infos.append(file_info[0])
            
            # Log the time taken to extract data from files
            logging.info('--- %s seconds to extract data from files ---' % (time.time() - start_time))
            
            if all_docs:
                # Store the extracted text as vectors
                file_results = store_vector(all_docs, fingerprint, True, file_infos)
                logging.info('--- %s seconds to create vector indexes ---' % (time.time() - start_time))
            
            # Create document summaries asynchronously
            threading.Thread(target=create_abstractive_summary, args=(fingerprint,)).start()
        except Exception as e:
            logging.error(f'Error processing document files: {e}')
            return jsonify({'message': f'Error processing files: {str(e)}'}), 500
    
        # Prepare response message
        message = "Files successfully uploaded."
    else:
        message = "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
    
    # Return success response
    response_data = {
        "status": "ok", 
        "message": message
    }
    
    # Include file details if available
    if file_results:
        successful_files = sum(1 for r in file_results if r.get('success', False))
        response_data["files_processed"] = len(file_results)
        response_data["files_successful"] = successful_files
        response_data["file_details"] = file_results
    
    return jsonify(response_data), 200

@app.route('/upload', methods=['POST', 'OPTIONS'])
def index(): 
    start_time = time.time()  # Record the start time for performance logging
    try:
        # Retrieve the token from the request form
        token = request.form.get('token')
        if not token:
            # Return an error response if the token is missing
            return jsonify({'message': 'Token is missing!'}), 401
        
        # Authenticate the user using the provided token
        user_session = authenticate(token)
        
        # Retrieve the session ID from the request form and append it to the user session
        session_id = request.form.get('sessionId')
        user_session = user_session + str(session_id.lower())
    except ExpiredSignatureError:
        # Handle the case where the token has expired
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        # Handle the case where the token is invalid
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        # Log any other exceptions that occur during token processing
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Log the time taken to authenticate the request
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))
    
    # Retrieve the list of files from the request
    files = request.files.getlist("files")
    logging.info(f'Request is \n {request.files}')
 
    # Separate files into valid document types and CSV files
    valid_files = [f for f in files if f.filename.endswith(('.pdf', '.docx', '.txt'))]
    csv_files = [f for f in files if f.filename.endswith(('.xlsx', '.csv'))]
    
    # Check if any files were uploaded
    if not files or files[0].filename == '':
        message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
        return jsonify({'message': message}), 400
    
    # Process CSV files if any
    if csv_files:
        # Add CSV files to the database
        success, message = create_database_with_tables(user_session, csv_files)
        if not success:
            return jsonify({"error": message}), 500
        logging.info('CSV files successfully uploaded!')
        # Create summaries asynchronously
        for file in csv_files:
            # Extract text from the file
            file_name = file.filename
            threading.Thread(target=store_table_info, args=(user_session, file_name,)).start()
            
    # Process valid document files
    file_results = []
    if valid_files:
        try:
            # Process each file individually
            all_docs = []
            file_infos = []
            
            for file in valid_files:
                # Extract text from the file
                docs, file_info = get_text_from_files([file])
                if docs:
                    all_docs.append(docs)
                    file_infos.append(file_info[0])
            
            # Log the time taken to extract data from files
            logging.info('--- %s seconds to extract data from files ---' % (time.time() - start_time))
            
            if all_docs:
                # Store the extracted text as vectors
                file_results = store_vector(all_docs, user_session, True, file_infos)
                logging.info('--- %s seconds to create vector indexes ---' % (time.time() - start_time))
            
            # Create document summaries asynchronously
            threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
        except Exception as e:
            logging.error(f'Error processing document files: {e}')
            return jsonify({'message': f'Error processing files: {str(e)}'}), 500
    
    # Prepare response message
    message = "Files successfully uploaded."
    if not valid_files and not csv_files:
        message = "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
    
    # Return success response
    response_data = {
        "status": "ok", 
        "message": message
    }
    
    # Include file details if available
    if file_results:
        successful_files = sum(1 for r in file_results if r.get('success', False))
        response_data["files_processed"] = len(file_results)
        response_data["files_successful"] = successful_files
        response_data["file_details"] = file_results
    
    return jsonify(response_data), 200


@app.route('/add-upload', methods=['POST', 'OPTIONS'])
def newfile():
    """
    Handle additional file uploads for authenticated users.
    
    This endpoint processes each file individually:
    1. Authenticates the user using a JWT token
    2. Processes each uploaded file separately
    3. Adds the extracted text to existing vector embeddings
    4. Creates file-specific summaries asynchronously
    
    Returns:
        JSON response with status, message, and file details
    """
    start_time = time.time()
    
    # Authenticate user
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_session = authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_session + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    logging.info('--- %s seconds to authenticate req ---' % (time.time() - start_time))

    # Retrieve the list of files from the request
    files = request.files.getlist("files")
    logging.info(f'Request is \n {request.files}')
    
    # Check if any files were uploaded
    if not files or files[0].filename == '':
        message = "Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
        return jsonify({'message': message}), 400
    
    # Separate files into valid document types and CSV files
    valid_files = [f for f in files if f.filename.endswith(('.pdf', '.docx', '.txt'))]
    csv_files = [f for f in files if f.filename.endswith(('.xlsx', '.csv'))]
    
    # Process CSV files if any
    if csv_files:
        # Add CSV files to the database
        success, message = add_tables_to_existing_db(user_session, csv_files)
        if not success:
            return jsonify({"error": message}), 400
        logging.info('CSV files successfully uploaded!')
        # Create summaries asynchronously
        for file in csv_files:
            # Extract text from the file
            file_name = file.filename
            threading.Thread(target=store_table_info, args=(user_session, file_name,)).start()
    
    # Process valid document files
    file_results = []
    if valid_files:
        try:
            # Process each file individually
            all_docs = []
            file_infos = []
            
            for file in valid_files:
                # Extract text from the file
                docs, file_info = get_text_from_files([file])
                if docs:
                    all_docs.append(docs)
                    file_infos.append(file_info[0])
            
            # Log the time taken to extract data from files
            logging.info('--- %s seconds to extract data from files ---' % (time.time() - start_time))
            
            if all_docs:
                # Store the extracted text as vectors, preserving existing data
                file_results = store_vector(all_docs, user_session, False, file_infos)
                logging.info('--- %s seconds to create vector indexes ---' % (time.time() - start_time))
            
            # Create document summaries asynchronously
            threading.Thread(target=create_abstractive_summary, args=(user_session,)).start()
        except Exception as e:
            logging.error(f'Error processing document files: {e}')
            return jsonify({'message': f'Error processing files: {str(e)}'}), 500
    
    # Prepare response message
    message = "Files successfully uploaded."
    if not valid_files and not csv_files:
        message = "No valid files were found. Please upload files in PDF, DOC, DOCX, TXT, CSV, or XLSX format."
    
    # Return success response
    response_data = {
        "status": "ok", 
        "message": message
    }
    
    # Include file details if available
    if file_results:
        successful_files = sum(1 for r in file_results if r.get('success', False))
        response_data["files_processed"] = len(file_results)
        response_data["files_successful"] = successful_files
        response_data["file_details"] = file_results
    
    return jsonify(response_data), 200

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Simple health check endpoint to verify the application is running.
    
    Returns:
        JSON response with status message
    """
    return {"message": "app is up and running"}

@app.route('/trialAsk', methods=['POST'])
@limiter.limit("100 per minute")
def trialAsk():
    """
    Handle document queries for users in free trial mode.
    
    This endpoint:
    1. Validates the user's fingerprint and trial limits
    2. Processes the user's question
    3. Generates a response using vector search and LLMs
    
    Returns:
        JSON response with answer
    """
    start_time = time.time()
    data = request.get_json()

    # Check free trial limits
    try:
        fingerprint = data.get('fingerprint')
        if is_trial_limit_over(fingerprint):
            return jsonify({'message': 'Free Trial limit is exhausted'}), 200
    except Exception as e:
        logging.info(f'Error validating fingerprint: {e}')
        return jsonify({'message': 'Error validating fingerprint'}), 400
    
    # Extract query parameters
    try:
        user_query = data.get('message')
        input_language = int(data.get('inputLanguage'))
        output_language = int(data.get('outputLanguage'))
        context = data.get('context')
        hascsvxl = data.get('hasCsvOrXlsx')
        mode = data.get('mode')
    except Exception as e:
        logging.info(e)
        return jsonify({'message': str(e)}), 400
    
    # Generate response using LLM
    try:
        logging.info(f'User query: {user_query}')
        session_name = fingerprint
        
        if context:
            llm_response = get_llm_response(
                user_query, input_language, output_language, 
                session_name, hascsvxl=hascsvxl, mode=mode
            )
        else:
            llm_response = get_general_llm_response(
                user_query, input_language, output_language, session_name
            )
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info(f'trialAsk response: {llm_response}')
    logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
    return jsonify(llm_response)

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle document queries for authenticated users.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Checks if the user has exceeded query limits
    3. Processes the user's question
    4. Generates a response using vector search and LLMs
    
    Returns:
        JSON response with answer
    """
    start_time = time.time()
    data = request.get_json()
    
    # Authenticate user
    session_name = ''
    try:
        token = data.get('token')
        logging.info("Token : " + str(token))
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        session_name = authenticate(token)
        session_id = data.get('sessionId')
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    logging.debug('here, after token')

    # Check if the user has exceeded query limits
    if is_user_limit_over(session_name):
        return jsonify({"answer": "To ask further questions, please upgrade your account."})
    
    # Extract query parameters
    try:
        user_query = data.get('message')
        input_language = int(data.get('inputLanguage'))
        output_language = int(data.get('outputLanguage'))
        context = data.get('context')
        hascsvxl = data.get('hasCsvOrXlsx')
        mode = data.get('mode')
        filenames = data.get('filenames', [])
    except Exception as e:
        logging.info(e)
        return jsonify({'message': str(e)}), 400
    
    # Generate response using LLM
    try:
        logging.info(f'User query: {user_query}')

        # Input guardrail processing
        input_guardrail = input_guardrail_pipeline()
        guardrail_response = input_guardrail.process_input(user_query)
        if guardrail_response.get("status") == "blocked":
            # If the input is blocked, return the guardrail response
            return jsonify(guardrail_response), 400
        else:
            # If the input is allowed, proceed with the LLM response
            user_query = guardrail_response.get("sanitized_input")
            logging.info(f'Sanitized user query: {user_query}')
        
        if context:
            session_name = session_name + str(session_id.lower())
            llm_response = get_llm_response(
                user_query, input_language, output_language, 
                session_name, hascsvxl=hascsvxl, mode=mode, filenames=filenames
            )
        else:
            llm_response = get_general_llm_response(
                user_query, input_language, output_language, session_name
            )
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
    return jsonify(llm_response)

@app.route('/demo', methods=['POST'])
@limiter.limit("10 per minute")
def transportdemo():
    """
    Handle demo queries for public transport information.
    
    This endpoint:
    1. Processes the user's question about public transport
    2. Generates a response using a pre-defined database
    
    Returns:
        JSON response with answer
    """
    start_time = time.time()
    
    # Extract query parameters
    data = request.get_json()
    try:
        user_query = data.get('message')
    except Exception as e:
        logging.info(e)
        return jsonify({'message': str(e)}), 400
    
    # Generate response using LLM
    try:
        logging.info(f'User query: {user_query}')
        session_name = "user20250317t114507"
        llm_response = get_demo_response(user_query, session_name)
    except Exception as e:
        logging.info(f'Error: {e}')
        return jsonify({'message': 'Error generating response from LLM'}), 500

    logging.info(f'Demo response: {llm_response}')
    logging.info('--- %s seconds to complete query response ---' % (time.time() - start_time))
    return jsonify({"answer": llm_response})

@app.route('/delete-container', methods=['POST'])
def delete_container():
    """
    Delete a user's container and associated data.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Deletes the user's container from the database
    
    Returns:
        JSON response with status message
    """
    # Extract request parameters
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        session_name = authenticate(token)
        session_id = request.form.get('sessionId')
        session_name = session_name + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logging.exception(f'Other error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Delete the user's container from the database
    try:
        delete_session(session_name)
    except Exception as e:
        logging.info(f'Error deleting container: {e}')
        return jsonify({'message': 'Error deleting container'}), 400

@app.route('/updatepayment', methods=['POST'])
def update_email():
    """
    Update a user's payment plan.
    
    This endpoint:
    1. Extracts the user's email and payment plan
    2. Calculates the plan duration based on the plan type
    3. Updates the user's account in the database
    
    Returns:
        JSON response with status message
    """
    # Extract request parameters
    try:
        data = request.get_json()
        email = data.get('email')
        plan = int(data.get('paymentPlan'))
    except Exception as e:
        logging.info(e)
        return jsonify({'message': 'Error extracting request data'}), 500
    
    if email and plan:
        # Determine plan duration
        if plan == 1:
            plan_limit_days = 30
        elif plan == 2:
            plan_limit_days = 90
        elif plan == 3:
            plan_limit_days = 365
        else:
            return jsonify({'message': 'Plan not supported'}), 400

        # Update account in database
        try:
            update_count = upgrade_account(email, plan_limit_days)
            if update_count:
                return jsonify({'message': 'Account upgraded successfully!'}), 200
            else:
                return jsonify({'message': 'Account with email not found!'}), 400
        except Exception as e:
            logging.info(f'Error upgrading account: {e}')
            return jsonify({'message': 'Error updating database'}), 400
    else:
        return jsonify({'message': 'Either email or plan empty!'}), 500

@app.route('/check-payment-status', methods=['POST'])
def check_payment_status():
    """
    Check a user's payment status.
    
    This endpoint:
    1. Extracts the user's email
    2. Checks the user's account status in the database
    3. Returns the status and remaining days
    
    Returns:
        JSON response with status and remaining days
    """
    # Extract request parameters
    try:
        data = request.get_json()
        email = data.get('email')
    except Exception as e:
        logging.info(e)
        return jsonify({'message': 'Error extracting request data'}), 500
    
    if email:
        try:
            # Get account status from database
            account_limit = get_account_status(email)
        except Exception as e:
            logging.info(f'Error checking user limits: {e}')
        
        # Determine status based on account limit
        status = 'not paid'
        if account_limit == 0:
            status = 'Expired plan. Please Renew'
        elif account_limit > 0:
            status = 'paid'
        
        return jsonify({'status': status, 'remaining_days': account_limit}), 200
    else:
        return jsonify({'message': 'Invalid email'}), 500

# Register webhook blueprint
app.register_blueprint(webhook_blueprint, url_prefix="/api")

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=5000, 
        ssl_context=(
            '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/fullchain.pem', 
            '/etc/letsencrypt/live/qdocbackend.carnotresearch.com/privkey.pem'
        )
    )