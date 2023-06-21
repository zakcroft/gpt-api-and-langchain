import os
import openai
from dotenv import load_dotenv, find_dotenv

# Schema
# from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Prompts
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Modals
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Chains
from langchain.chains import LLMChain


load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]


template = """
    Interprete the text and evaluate the text.
    sentiment: is the text in a positive, neutral or negative sentiment?
    subject: What subject is the text about? Use exactly one word.

    Format the output as JSON with the following keys:
    sentiment
    subject

    text: {input}
"""

# chat = ChatOpenAI(temperature=0)

llm = OpenAI(model_name="text-ada-001")

one_prompt_template = PromptTemplate.from_template(template=template)
chain = LLMChain(llm=llm, prompt=one_prompt_template)

output = chain.predict(input="I love France")

print('1', output)

two_prompt_template = ChatPromptTemplate.from_template(template=template)
llmChat = ChatOpenAI(model_name="gpt-3.5-turbo")

chainChat = LLMChain(llm=llmChat, prompt=two_prompt_template)
outputChat = chainChat.predict(
    input="I ordered Pizza Salami and it was awesome!")
print('2', outputChat)
