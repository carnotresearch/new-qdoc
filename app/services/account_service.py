"""
Account management service.

This module handles user account management, payment processing, and subscription status.
"""

import logging
from typing import Dict, Any, Tuple

from controllers.database import upgrade_account, get_account_status

logger = logging.getLogger(__name__)

class AccountService:
    """Service for handling user accounts and payments."""
    
    def update_payment_plan(self, email: str, plan: int) -> Tuple[Dict[str, Any], int]:
        """
        Update a user's payment plan.
        
        Args:
            email: User's email address
            plan: Payment plan identifier (1=Monthly, 2=Quarterly, 3=Yearly)
            
        Returns:
            Tuple[Dict[str, Any], int]: Response data and HTTP status code
        """
        if not email or not plan:
            return {'message': 'Email or plan missing'}, 400
            
        # Determine plan duration
        if plan == 1:
            plan_limit_days = 30  # Monthly
        elif plan == 2:
            plan_limit_days = 90  # Quarterly
        elif plan == 3:
            plan_limit_days = 365  # Yearly
        else:
            return {'message': 'Plan not supported'}, 400

        # Update account in database
        try:
            update_count = upgrade_account(email, plan_limit_days)
            if update_count:
                return {'message': 'Account upgraded successfully!'}, 200
            else:
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
        if not email:
            return {'message': 'Invalid email'}, 400
        
        try:
            # Get account status from database
            account_limit = get_account_status(email)
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
account_service = AccountService()

def get_account_service() -> AccountService:
    """
    Get the account service singleton instance.
    
    Returns:
        AccountService: The account service instance
    """
    return account_service