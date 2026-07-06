import os
from dotenv import load_dotenv
load_dotenv()

from bq_client import BigQueryRunner
runner = BigQueryRunner(project_id=os.environ.get("GCP_PROJECT_ID"))
df = runner.execute_query(
    "SELECT COUNT(*) AS n FROM `bigquery-public-data.thelook_ecommerce.orders`"
)
print("BigQuery OK — orders rows:", df["n"][0])

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash",
                             google_api_key=os.environ["GOOGLE_API_KEY"])
print("Gemini OK —", llm.invoke("Say 'ready' and nothing else.").content)