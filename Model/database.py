from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_database():
    # Get the MongoDB connection string from the environment variable
    connection_string = os.getenv('MONGODB_URI')

    if not connection_string:
        raise Exception("MongoDB connection string is not set in the environment variables.")

    # Create a connection using MongoClient
    client = MongoClient(connection_string)

    # The ismaster command is cheap and does not require auth.
    server_info = client.admin.command('ismaster')

    # Check if the server_info call does not throw an exception
    if server_info:
        print(f"Connected to database: {server_info['ismaster']}")

    # Return the database instance
    return client['face_recognition_db']