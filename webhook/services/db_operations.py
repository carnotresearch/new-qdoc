from pymongo import MongoClient

# Initialize MongoDB connection
client = MongoClient("mongodb+srv://icarkno:Hxgg7HTN0WsejGSA@icarkno.j91m9.mongodb.net/?retryWrites=true&w=majority&appName=icarKno")
db = client.get_database("test")  # Use default database from URI
message_collection = db['message']  # Access the 'message' collection

def check_or_add_message_id(message_id: str) -> bool:
    """
    Check if an ID exists in the 'message' collection.
    If it exists, return True.
    If it doesn't exist, add it with a status of 1 and return False.
    """
    existing_entry = message_collection.find_one({"_id": message_id})
    if existing_entry:
        return True  # ID already exists in the collection
    
    # Insert new entry with status: 1
    message_collection.insert_one({"_id": message_id, "status": 1})
    return False