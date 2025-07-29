import psycopg2
import os
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
from urllib.parse import quote_plus


# Fetch individual parameters from the environment
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

encoded_password = quote_plus(PASSWORD)
connection_string = f"postgresql://{USER}:{encoded_password}@{HOST}:{PORT}/{DBNAME}"

# Ensure all parameters are loaded correctly
if None in [USER, PASSWORD, HOST, PORT, DBNAME]:
    raise ValueError("One or more database connection parameters are missing in the .env file")

# Connect to the database using individual parameters
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")

except Exception as e:
    print(f"Failed to connect: {e}")
