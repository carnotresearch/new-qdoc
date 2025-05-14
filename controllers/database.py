"""
Database operations module for MongoDB interactions.

This module provides functions for:
- User authentication and management
- Trial status checking
- Database connection and operations
"""

# Standard library imports
import logging
from datetime import date, datetime, timedelta

# Third-party imports
from pymongo import MongoClient
from config import Config

# MongoDB configuration
mongo_url = Config.MONGO_URL
client = MongoClient(mongo_url)
db = client.test
collection = db.users


def check_trial_status(fingerprint):
    """
    Check if a user is eligible for a free trial.
    
    Args:
        fingerprint (str): User's fingerprint identifier
        
    Returns:
        bool: True if the user is eligible for a free trial, False otherwise
    """
    condition = {"fingerprint": str(fingerprint)}
    fingerprint_collection = db.fingerprint
    fingerprint_detail = fingerprint_collection.find_one(condition)
    
    if fingerprint_detail and fingerprint_detail.get('freeTrial') == 'true':
        return False

    # Add entry to fingerprint collection
    new_entry = {
        "fingerprint": str(fingerprint),
        "freeTrial": "true",
        "message": 0
    }
    fingerprint_collection.insert_one(new_entry)

    return True


def create_fingerprint_entry(fingerprint):
    """
    Create or reset a fingerprint entry in the database.
    
    Args:
        fingerprint (str): User's fingerprint identifier
    """
    fingerprint_collection = db.fingerprint
    condition = {"fingerprint": str(fingerprint)}

    # Check if the fingerprint already exists
    if fingerprint_collection.find_one(condition):
        # Reset message count to 0 if the entry exists
        fingerprint_collection.update_one(condition, {"$set": {"message": 0}})
        return

    # Create a new entry if it does not exist
    fingerprint_collection.insert_one({
        "fingerprint": str(fingerprint),
        "message": 0
    })


def is_trial_limit_over(fingerprint):
    """
    Check if a user has exceeded the free trial message limit.
    
    Args:
        fingerprint (str): User's fingerprint identifier
        
    Returns:
        bool: True if the user has exceeded the trial limit, False otherwise
    """
    trial_message_limit = 10
    condition = {"fingerprint": str(fingerprint)}
    fingerprint_collection = db.fingerprint
    fingerprint_detail = fingerprint_collection.find_one(condition)

    if fingerprint_detail and fingerprint_detail.get('message', 0) >= trial_message_limit:
        return True

    # Increment the 'message' field
    fingerprint_collection.update_one(
        condition,
        {"$inc": {"message": 1}}
    )

    return False


def is_user_limit_over(session_name):
    """
    Check if a user has exceeded their query limit.
    
    Args:
        session_name (str): User's session identifier
        
    Returns:
        bool: True if the user has exceeded their limit, False otherwise
    """
    condition = {"email": str(session_name)}
    user_details = collection.find_one(condition)
    
    paid = 0
    if user_details:
        if user_details.get('paid'):
            paid = int(user_details.get('paid'))
    else:
        # User not found (Illegal login)
        return True

    if paid == 0:
        queries = int(user_details.get('queries'))
        if queries < 10:
            # Increment the queries count
            result = collection.update_one(
                condition, 
                {"$inc": {"queries": 1}}
            )
            logging.info(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")
        elif queries >= 10:
           return True

    return False


def upgrade_account(email, plan_limit_days):
    """
    Upgrade a user's account with a new payment plan.
    
    Args:
        email (str): User's email address
        plan_limit_days (int): Number of days for the plan
        
    Returns:
        int: Number of documents modified (1 if successful, 0 if not found)
    """
    # Calculate expiry date
    expiry_date = date.today() + timedelta(days=plan_limit_days)

    # Convert to YYYYMMDD format
    expiry_date_int = int(expiry_date.strftime("%Y%m%d"))
    logging.info(f"Expiry date (YYYYMMDD): {expiry_date_int}")

    # Find the document with the given email and update it
    result = collection.update_one(
        {'email': email},
        {'$set': {'paid': 1, 'expiry_date': expiry_date_int}}
    )

    return result.modified_count


def get_account_status(email):
    """
    Get a user's account status and remaining days.
    
    Args:
        email (str): User's email address
        
    Returns:
        int: Number of days remaining (-1 if not paid)
    """
    # Get user from database
    user = collection.find_one({'email': email})
    
    if user:
        payment_status = user.get('paid', 0)
        expiry_date_int = user.get('expiry_date', 0)  # Get expiry_date as int

        expiry_date_str = str(expiry_date_int)  # Convert int to string
        expiry_date = datetime.strptime(expiry_date_str, "%Y%m%d").date()
        remaining_days = (expiry_date - date.today()).days
        logging.info(f'Remaining days for user: {remaining_days}')

        status = 'paid' if payment_status != 0 else 'not paid'
        if status == 'paid':
            return remaining_days

    return -1