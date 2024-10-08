import os
from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from .env file
print(os.getenv('IP_ADDRESS'))  # Should now print 192.168.1.3