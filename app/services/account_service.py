"""
Account management service.

This module handles user account management, payment processing, and subscription status.
"""

import logging
from typing import Dict, Any, Tuple

from controllers.database import upgrade_account, get_account_status

# Configure logging
logger = logging.getLogger(__name__)

class AccountService:
    """Service for handling user accounts and payments."""
    
    def __init__(self):
        """Initialize account service."""
        logger.info("Initializing account service")
        self.plan_durations = {
            1: 30,   # Monthly
            2: 90,   # Quarterly
            3: 365   # Yearly
        }
    
    def update_payment_plan(self, email: str, plan: int) -> Tuple[Dict[str, Any], int]:
        """
        Update a user's payment plan.
        
        Args:
            email: User's email address
            plan: Payment plan identifier (1=Monthly, 2=Quarterly, 3=Yearly)
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        logger.info(f"Updating payment plan for user {email} to plan {plan}")
        
        # Validate parameters
        if not email or not plan:
            logger.warning("Missing email or plan in update request")
            return {'message': 'Email or plan missing'}, 400
            
        # Determine plan duration
        if plan not in self.plan_durations:
            logger.warning(f"Unsupported plan type: {plan}")
            return {'message': 'Plan not supported'}, 400
            
        plan_limit_days = self.plan_durations[plan]
        logger.info(f"Plan duration: {plan_limit_days} days")

        # Update account in database
        try:
            update_count = upgrade_account(email, plan_limit_days)
            if update_count:
                logger.info(f"Successfully upgraded account for {email}")
                return {'message': 'Account upgraded successfully!'}, 200
            else:
                logger.warning(f"Account not found for email: {email}")
                return {'message': 'Account with email not found!'}, 404
        except Exception as e:
            logger.exception(f'Error upgrading account: {e}')
            return {'message': 'Error updating database'}, 500
    
    def get_payment_status(self, email: str) -> Tuple[Dict[str, Any], int]:
        """
        Check a user's payment status.
        
        Args:
            email: User's email address
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data (status and remaining days) and HTTP status code
        """
        logger.info(f"Checking payment status for user {email}")
        
        if not email:
            logger.warning("Invalid email in payment status check")
            return {'message': 'Invalid email'}, 400
        
        try:
            # Get account status from database
            account_limit = get_account_status(email)
            logger.info(f"Account limit for {email}: {account_limit} days")
        except Exception as e:
            logger.exception(f'Error checking user limits: {e}')
            return {'message': 'Error checking account status'}, 500
        
        # Determine status based on account limit
        status = 'not paid'
        if account_limit == 0:
            status = 'Expired plan. Please Renew'
        elif account_limit > 0:
            status = 'paid'
        
        return {'status': status, 'remaining_days': account_limit}, 200

# Create a singleton instance
_account_service = None

def get_account_service() -> AccountService:
    """
    Get the account service singleton instance.
    
    Returns:
        AccountService: The account service instance
    """
    global _account_service
    if _account_service is None:
        _account_service = AccountService()
    return _account_service