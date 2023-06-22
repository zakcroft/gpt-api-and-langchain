import os
import openai
import json
from dotenv import load_dotenv, find_dotenv

from _setup import print_to_pretty_json

# Schema
# from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Prompts
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Modals
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Chains
from langchain.chains import LLMChain, SimpleSequentialChain

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

# llm = OpenAI(model_name="text-davinci-001")
llmChat = ChatOpenAI(model_name="gpt-3.5-turbo")

# one_prompt_template = PromptTemplate.from_template(template=template)
# print_to_pretty_json(one_prompt_template, 'one_prompt_template')
# chain = LLMChain(llm=llm, prompt=one_prompt_template)

# output = chain.predict(input="I love France")
# print('1', output)

two_prompt_template = ChatPromptTemplate.from_template(template=template)
# print_to_pretty_json(two_prompt_template, 'two_prompt_template')

chainChat = LLMChain(llm=llmChat, prompt=two_prompt_template)
# outputChat = chainChat.predict(
#     input="I ordered Pizza Salami and it was awesome!")
# print_to_pretty_json(chainChat, 'chainChat')

response_template = """
You are a helpful bot that creates a 'thank you' response text.
If customers are unsatisfied, offer them a real world assistant to talk to.
You will get a sentiment and subject as into and evaluate.

text: {input}
"""
review_template = ChatPromptTemplate.from_template(template=response_template)
chainReview = LLMChain(llm=llmChat, prompt=review_template)
print_to_pretty_json(review_template, 'review_template')
print_to_pretty_json(chainReview, 'review_chain')

overall_chain = SimpleSequentialChain(chains=[chainChat, chainReview], verbose=True)
#
overall_chain.run(input="I ordered Pizza Salami and was nice!")
overall_chain.run(input="I ordered Pizza Salami and was aweful!")
