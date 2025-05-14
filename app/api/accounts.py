"""
Account management API routes.

This module defines routes for account management, payment processing, and subscription status.
"""

import logging
from flask import Blueprint, request, jsonify

from app.services.account_service import get_account_service

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
accounts_bp = Blueprint('accounts', __name__)

# Get service instance
account_service = get_account_service()

@accounts_bp.route('/updatepayment', methods=['POST'])
def update_payment():
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
        logger.exception(f'Invalid request format: {e}')
        return jsonify({'message': 'Error extracting request data'}), 500
    
    # Update payment plan
    response, status_code = account_service.update_payment_plan(email, plan)
    return jsonify(response), status_code

@accounts_bp.route('/check-payment-status', methods=['POST'])
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
        logger.exception(f'Invalid request format: {e}')
        return jsonify({'message': 'Error extracting request data'}), 500
    
    # Get payment status
    response, status_code = account_service.get_payment_status(email)
    return jsonify(response), status_code