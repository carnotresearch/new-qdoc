"""
Document management API routes.

This module defines routes for document upload, processing, and container management.
"""

import logging
from flask import Blueprint, request, jsonify
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import re
import json
import os
from app.services.auth_service import get_auth_service
from app.services.document_service import get_document_service
from app.services.create_dense_graph_service import process_text_file_dense
from app.services.create_highlevel_graph_service import process_text_file
from utils.neo4j_helper import save_neo4j_graph, fetch_neo4j_graph_data
import threading
from config import Config


# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
documents_bp = Blueprint('documents', __name__)

# Get service instances
auth_service = get_auth_service()
document_service = get_document_service()

# ----------------------------------------
# Free Trial Routes
# ----------------------------------------

@documents_bp.route('/freeTrial', methods=['POST', 'OPTIONS'])
def free_trial(): 
    """
    Handle file uploads for users in free trial mode.
    
    This endpoint processes files for users without authentication:
    1. Validates the user's fingerprint
    2. Processes uploaded files
    3. Stores the extracted text as vector embeddings
    4. Creates summaries asynchronously
    
    Returns:
        JSON response with status, message, and file details
    """
    try:
        # Extract and validate fingerprint
        fingerprint = request.form.get('fingerprint')
        if not fingerprint:
            return jsonify({'message': 'Fingerprint is missing'}), 400
            
        # Process files
        result = document_service.process_files(
            request.files, fingerprint, is_new_container=True, is_trial=True
        )
        
        if result.get("status") == "error":
            return jsonify({'message': result.get("message")}), 400
            
        return jsonify(result), 200
    except Exception as e:
        logger.exception(f'Error in free trial: {e}')
        return jsonify({'message': f'Error processing files: {str(e)}'}), 500

# ----------------------------------------
# Authenticated User Routes
# ----------------------------------------

@documents_bp.route('/upload', methods=['POST', 'OPTIONS'])
def upload(): 
    """
    Handle file uploads for authenticated users to create a new container.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Processes uploaded files
    3. Creates a new document container
    
    Returns:
        JSON response with status, message, and file details
    """
    # Authenticate user
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Process files
    result = document_service.process_files(
        request.files, user_session, is_new_container=True, is_trial=False
    )
    
    if result.get("status") == "error":
        return jsonify({'message': result.get("message")}), 400
        
    return jsonify(result), 200

@documents_bp.route('/add-upload', methods=['POST', 'OPTIONS'])
def add_upload():
    """
    Handle additional file uploads for authenticated users.
    
    This endpoint:
    1. Authenticates the user using a JWT token
    2. Processes uploaded files
    3. Adds to an existing document container
    
    Returns:
        JSON response with status, message, and file details
    """
    # Authenticate user
    try:
        token = request.form.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Process files
    result = document_service.process_files(
        request.files, user_session, is_new_container=False, is_trial=False
    )
    
    if result.get("status") == "error":
        return jsonify({'message': result.get("message")}), 400
        
    return jsonify(result), 200

@documents_bp.route('/delete-container', methods=['POST'])
def delete_container_route():
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
        
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')
        user_session = user_email + str(session_id.lower())
    except ExpiredSignatureError:
        return jsonify({'message': 'Token has expired!'}), 401
    except InvalidTokenError as e:
        return jsonify({'message': 'Token is invalid!'}), 401
    except Exception as e:
        logger.exception(f'Authentication error: {e}')
        return jsonify({'message': 'Token decoding failed!'}), 401
    
    # Delete the user's container from the database
    success = document_service.delete_container(user_session)
    if success:
        return jsonify({'message': 'Container deleted successfully'}), 200
    else:
        return jsonify({'message': 'Error deleting container'}), 500
    

@documents_bp.route('/create_graph', methods=['POST'])
def create_knowledge_graph():
    try:
        token = request.form.get('token')
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')

        if not user_email:
            return jsonify({'message': 'Authentication Failed'}), 401
        
        # Start async processing
        file_count = 0
        file_names = []
        while f'file_{file_count}' in request.form:
            file_names.append(request.form.get(f'file_{file_count}'))
            file_count += 1

        user_session = user_email + str(session_id.lower())
        output_dir = os.path.join("users", user_session, "files")
        
        status_file = os.path.join(output_dir, "graph_processing.txt")
        with open(status_file, 'w') as f:
            f.write("processing")

        # Start async processing
        is_dense = request.form.get('isDense', 'false').lower() == 'true'
        threading.Thread(
            target=async_process_graph,
            args=(file_names, user_session, session_id, output_dir, is_dense)
        ).start()

        return jsonify({
            'message': 'Knowledge graph creation started',
            'status': 'processing'
        }), 202

    except Exception as e:
        logger.exception(f"Error in knowledge graph creation: {e}")
        return jsonify({
            'message': 'Failed to start knowledge graph creation',
            'error': str(e)
        }), 500

def async_process_graph(file_names, user_session, session_id, output_dir, is_dense):
    try:
        for file_name in file_names:
            file_name = re.sub(r'\.[^.\\/:*?"<>|\r\n]+$', '', file_name)
            content_path = os.path.join("users", user_session, "files", file_name, "chunks_content.txt")
            
            if os.path.exists(content_path):
                if is_dense:
                    process_text_file_dense(content_path, session_id, file_name)
                else:
                    process_text_file(content_path, session_id, file_name)
        
        graph_data = fetch_neo4j_graph_data(
            session_id=session_id,
            neo4j_uri=Config.NEO4J_URI,
            neo4j_username=Config.NEO4J_USERNAME,
            neo4j_password=Config.NEO4J_PASSWORD
        )

        output_path = os.path.join(output_dir, "graph.json")
        save_neo4j_graph(graph_data, output_path)

        with open(os.path.join(output_dir, "graph_processing.txt"), 'w') as f:
            f.write("completed")

    except Exception as e:
        logger.exception(f"Error in async graph processing: {e}")
        with open(os.path.join(output_dir, "graph_processing.txt"), 'w') as f:
            f.write(f"error: {str(e)}")



@documents_bp.route('/check_graph_status', methods=['POST'])
def check_graph_status():
    try:
        token = request.form.get('token')
        user_email = auth_service.authenticate(token)
        session_id = request.form.get('sessionId')

        if not user_email:
            return jsonify({'message': 'Authentication Failed'}), 401

        user_session = user_email + str(session_id.lower())

        output_dir = os.path.join("users", user_session, "files")
        status_file = os.path.join(output_dir, "graph_processing.txt")
        graph_file = os.path.join(output_dir, "graph.json")

        if not os.path.exists(status_file):
            return jsonify({
                'status': 'not_started',
                'message': 'Graph processing not started'
            }), 200

        with open(status_file, 'r') as f:
            status = f.read().strip()

        if status == 'completed' and os.path.exists(graph_file):
            return jsonify({
                'status': 'completed',
                'message': 'Knowledge graph created successfully',
                'graph_data_path': graph_file
            }), 200
        elif status.startswith('error'):
            return jsonify({
                'status': 'error',
                'message': status.split(':', 1)[1] if ':' in status else 'Error processing graph'
            }), 200
        else:
            return jsonify({
                'status': 'processing',
                'message': 'Knowledge graph creation in progress'
            }), 200

    except Exception as e:
        logger.exception(f"Error checking graph status: {e}")
        return jsonify({
            'message': 'Failed to check graph status',
            'error': str(e)
        }), 500 


@documents_bp.route('/fetch_graph', methods=['GET'])
def fetch_graph_data():
    try:
        session_id = request.args.get('sessionId')
        token = request.args.get('token')
        session_id = session_id.lower()
        user_email = auth_service.authenticate(token)

        if not user_email:
            logger.warning(f"Invalid token for session: {session_id}")
            return jsonify({"error": "Invalid token"}), 401
        user_session = user_email + str(session_id.lower())
        user_dir = os.path.join("users", user_session, "files", "graph.json")
        logger.info(f"Looking for graph at: {user_dir}")


        if not os.path.exists(user_dir):
            logger.error(f"Graph file not found at {user_dir}")
            return jsonify({"error": "Knowledge graph not found for this session"}), 404

        if not os.access(user_dir, os.R_OK):
            logger.error(f"File exists but cannot be read: {user_dir}")
            return jsonify({"error": "Cannot read knowledge graph file"}), 403

        file_size = os.path.getsize(user_dir)
        logger.info(f"Graph file size: {file_size} bytes")

        with open(user_dir, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            logger.debug(f"First 200 chars of file:\n{raw_content[:200]}...")
            try:
                graph_data = json.loads(raw_content)
                logger.info(f"Successfully parsed JSON with {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('relationships', []))} relationships")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed at line {e.lineno}, column {e.colno}: {e.msg}")
                logger.error(f"Problematic content: {raw_content[max(0, e.pos - 50):e.pos + 50]}")
                raise

        processed_nodes = []
        color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#7986CB']
        label_color_map = {}

        logger.info(f"Processing {len(graph_data.get('nodes', []))} nodes")

        for node in graph_data.get('nodes', []):
            if 'id' not in node:
                logger.warning(f"Skipping node missing 'id': {node}")
                continue

            labels = node.get('labels', [])
            if "Document" in labels:
                logger.info(f"Skipping Document node: {node.get('id')}")
                continue

            primary_label = next((l for l in labels if l != '_Entity_'), labels[0] if labels else 'default')

            if primary_label not in label_color_map:
                label_color_map[primary_label] = color_palette[len(label_color_map) % len(color_palette)]

            processed_nodes.append({
                "id": node["id"],
                "labels": labels,
                "name": node.get("name") or node.get("id") or "Unnamed Node",
                "color": label_color_map[primary_label],
                "size": 25,
                "caption": node.get("name") or node.get("id") or "Unnamed Node",
                "properties": node.get("properties", {})
            })

        logger.info(f"Processed {len(processed_nodes)} nodes")

        processed_relationships = []
        node_ids = {node['id'] for node in processed_nodes}
        logger.info(f"Processing {len(graph_data.get('relationships', []))} relationships")

        for rel in graph_data.get('relationships', []):
            from_id = rel.get('from') or rel.get('source')
            to_id = rel.get('to') or rel.get('target')

            if not from_id or not to_id:
                logger.warning(f"Skipping relationship missing endpoints: {rel}")
                continue
            if from_id not in node_ids or to_id not in node_ids:
                logger.warning(f"Skipping relationship with missing nodes: {from_id} -> {to_id}")
                continue

            processed_relationships.append({
                "id": rel.get('id') or f"{from_id}-{to_id}-{rel.get('type', 'REL')}",
                "from": from_id,
                "to": to_id,
                "type": rel.get('type') or 'RELATED',
                "caption": rel.get('caption') or rel.get('type') or 'related to',
                "color": rel.get('color') or '#a1a1a1',
                "width": rel.get('width') or 2,
                "properties": rel.get('properties', {})
            })

        logger.info(f"Processed {len(processed_relationships)} relationships")

        return jsonify({
            "nodes": processed_nodes,
            "relationships": processed_relationships,
            "message": "Successfully loaded graph data"
        })

    except Exception as e:
        logger.error(f"Unexpected error in fetch_graph_data: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500




