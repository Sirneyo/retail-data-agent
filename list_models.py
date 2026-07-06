import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)