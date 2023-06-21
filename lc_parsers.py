import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-ada-001")
output = llm("Tell me a joke")
print(output)
print(repr(output)) # with new lines
