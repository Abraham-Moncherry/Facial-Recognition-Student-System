import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to connect to PostgreSQL
def connect_db():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD"),
            host=os.getenv("DATABASE_HOST"),
            port=os.getenv("DATABASE_PORT"),
            options=os.getenv("DATABASE_OPTION")
        )
        print("✅ Database connection successful!")
        
        return conn
    except psycopg2.Error as e:
        print(f"❌ Database connection failed: {e}")
        return None

