import logging
import requests

def fetch_session_ids(token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IndoYXRzYXBwQGNhcm5vdHJlc2VhcmNoLmNvbSIsImlhdCI6MTczNDAwNjMwOH0.0Qt6iW3ptchU22ZDmbDQvnREKZll9xo_0BySRKCm0_0"):
    url = "https://2n5j71807b.execute-api.ap-south-1.amazonaws.com/default/fetchSessions"
    headers = {"Content-Type": "application/json"}
    payload = {"token": token}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json().get("data", {})
        sessions_data = data.get("sessions", [])
        
        if not sessions_data:
            logging.info("No sessions available.")
            return []
        
        # Extract session details
        sessions = [
            {
                "id": session.get("session_id"),
                "name": session.get("name", "Unnamed Session"),
            }
            for session in sessions_data
        ]
        
        return sessions
    
    except requests.RequestException as e:
        logging.error(f"Failed to fetch session IDs: {e}")
        return []