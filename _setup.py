import os
import openai
import json

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]

def print_to_pretty_json(input, description='json dump'):
    json_object = json.loads(input.json())
    print(description, json.dumps(json_object, indent=4))
