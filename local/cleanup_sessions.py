import os
import time
import logging

# Configure logging
logging.basicConfig(
    filename='/home/pranav/flask_backend_app/cleanup_sessions.log',  # Path to your log file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directory where your session folders are stored
sessions_dir = '/home/pranav/flask_backend_app'

# Time threshold in seconds (1 hour)
threshold = 4000

# Current time
now = time.time()

# Log script start
logging.info("Starting cleanup script")

# Iterate through the folders in the session directory
for folder_name in os.listdir(sessions_dir):
    folder_path = os.path.join(sessions_dir, folder_name)
    
    # Check if it's a directory and not a __pycache__ folder
    if os.path.isdir(folder_path) and folder_name != '__pycache__':
        # Get the last accessed time
        last_accessed = os.path.getatime(folder_path)
        
        # Check if the folder has not been accessed in the last hour
        if now - last_accessed > threshold:
            # Remove the folder and its contents
            try:
                os.system(f'rm -rf {folder_path}')
                logging.info(f"Deleted folder: {folder_path}")
            except Exception as e:
                logging.error(f"Failed to delete folder {folder_path}: {e}")

# Log script end
logging.info("Cleanup script finished")
