import logging
from flask import current_app, jsonify
import json
import requests
import yaml
import re
import os
from ..services.db_operations import check_or_add_message_id
from ..services.demo_service import get_demo_response

def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")

def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )

def send_message(data):
    headers = {
        "Content-type": "application/json",
        "Authorization": "Bearer EAANALnrVtSIBOz5ZAJBrsmNmK4ZBDzCRQEeVYsWRqA0aBnmKTwajA2TiXuEJxPlXrAWDeWJ4ZCr98Iw9qzWPwzrKIQOWZC8HyyGs5c407RlL6Uy0q1d8aVk8wmdnlQ1OEKh4lMzPR55zo4OSOZA7n7Yxcsz91gLYozWDcptsHZBHRq5xrGF36SbHw8pv2hrg9guQZDZD",#ACCESS_TOKEN
    }

    url = "https://graph.facebook.com/v18.0/531433223397578/messages"#PHONE_NUMBER_ID

    try:
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except (
        requests.RequestException
    ) as e:  # This will catch any general request exception
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        # Process the response as normal
        log_http_response(response)
        return response

def process_text_for_whatsapp(text):
    """
    Format text for WhatsApp display
    
    Args:
        text (str): The text to format
        
    Returns:
        str: Formatted text for WhatsApp
    """
    # Remove brackets
    pattern = r"\【.*?\】"
    text = re.sub(pattern, "", text).strip()

    # Convert markdown-style bold (**text**) to WhatsApp bold (*text*)
    pattern = r"\*\*(.*?)\*\*"
    replacement = r"*\1*"
    whatsapp_style_text = re.sub(pattern, replacement, text)
    
    # Handle emoji spacing (WhatsApp rendering can be odd with emojis)
    emoji_pattern = r'([\U00010000-\U0010ffff])'
    # Add spaces around emojis, but preserve existing newlines
    whatsapp_style_text = re.sub(emoji_pattern, r' \1 ', whatsapp_style_text)
    
    # Format numbered lists for metro stations or locations better
    list_pattern = r'(\d+)\.\s+(.*?)(?=\n\d+\.|\Z)'
    whatsapp_style_text = re.sub(list_pattern, r'*\1.* \2', whatsapp_style_text, flags=re.DOTALL)
    
    # Format landmark instructions better
    landmark_pattern = r'(Reply with the number or type \'landmark\' to provide a nearby landmark instead\.)'
    whatsapp_style_text = re.sub(landmark_pattern, r'_\1_', whatsapp_style_text)
    
    return whatsapp_style_text.strip() # Ensure no leading/trailing whitespace on the whole message

def split_message(text, max_length=4000):
    """
    Split long messages into multiple parts for WhatsApp
    
    Args:
        text (str): The message text to split
        max_length (int): Maximum message length
        
    Returns:
        list: List of message chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    lines = text.split('\n')
    
    for line in lines:
        # If adding this line would exceed max length, start a new chunk
        if len(current_chunk) + len(line) + 1 > max_length:
            # Only add the chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
        
        # Add the line to current chunk
        if current_chunk:
            current_chunk += '\n' + line
        else:
            current_chunk = line
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add part numbers if there are multiple chunks
    if len(chunks) > 1:
        for i in range(len(chunks)):
            chunks[i] = f"Part {i+1}/{len(chunks)}\n\n{chunks[i]}"
    
    return chunks

def process_whatsapp_message(body):
    """Process incoming WhatsApp messages and send responses"""
    logging.info(f"Processing WhatsApp message: {body}")
    
    # Extract message data
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    
    # Only process text messages
    if message.get("type", "text") == "text":
        message_body = message["text"]["body"]
        message_id = message["id"]
        
        # Check for duplicate messages
        exists = check_or_add_message_id(message_id)
        if exists:
            logging.info(f"Skipping duplicate message: {message_id}")
            return

        # Process message
        try:
            logging.info(f"Processing message from {name} ({wa_id}): {message_body}")

            logging.info("Using icarKno for response - not metro related")
            response = get_demo_response(message_body)
            
            # Format response for WhatsApp
            response = process_text_for_whatsapp(response)
            
            # Handle messages that are too long
            if len(response) > 4000:
                # Split into multiple messages
                chunks = split_message(response)
                for chunk in chunks:
                    data = get_text_message_input(f"+{wa_id}", chunk)
                    send_message(data)
            else:
                # Send as a single message
                data = get_text_message_input(f"+{wa_id}", response)
                send_message(data)
                
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")
            error_response = "I'm sorry, I encountered an issue. Please try again."
            data = get_text_message_input(f"+{wa_id}", error_response)
            send_message(data)
    else:
        # Non-text messages are not supported yet
        logging.info(f"Received non-text message type: {message.get('type')}")
        response = "I can only process text messages at the moment."
        data = get_text_message_input(f"+{wa_id}", response)
        send_message(data)

def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )