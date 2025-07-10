import os
from dotenv import load_dotenv

def validate_environment():
    load_dotenv()
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    print("✅ Environment variables validated successfully")
