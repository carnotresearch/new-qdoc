import requests
import logging

def get_demo_response(message):
    """Send POST request to demo URL."""
    # Load configuration from YAML
    url = "https://qdocbackend.carnotresearch.com/demo"
    headers = {
        "Content-Type": "application/json"
    }

    icarkno_data = {
        "message": message
    }
    
    # Send POST request
    response = requests.post(url, headers=headers, json=icarkno_data)
    
    if response.status_code == 200:
        logging.info("icarKno Response:", response.json())
    else:
        logging.info(f"icarKno req {response.status_code}: {response.text}")
    
    response_data = response.json()  # Extracts the JSON data as a dictionary
    return response_data.get('answer')  # Returns the 'answer' from the dictionary