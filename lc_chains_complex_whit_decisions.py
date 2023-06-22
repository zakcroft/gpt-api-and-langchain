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
from langchain.chains import LLMChain, SequentialChain

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-ada-001")

# This is an LLMChain to write a review given a dish name and the experience.
prompt_review = PromptTemplate.from_template(
    template="You ordered {dish_name} and your experience was {experience}. Write a review: "
)
chain_review = LLMChain(llm=llm, prompt=prompt_review, output_key="review")

# This is an LLMChain to write a follow-up comment given the restaurant review.
prompt_comment = PromptTemplate.from_template(
    template="Given the restaurant review: {review}, write a follow-up comment: "
)
chain_comment = LLMChain(llm=llm, prompt=prompt_comment, output_key="comment")

# This is an LLMChain to summarize a review.
prompt_summary = PromptTemplate.from_template(
    template="Summarise the review in one short sentence: \n\n {review}"
)
chain_summary = LLMChain(llm=llm, prompt=prompt_summary, output_key="summary")

# This is an LLMChain to translate a summary into German.
prompt_translation = PromptTemplate.from_template(
    template="Translate the summary to german: \n\n {summary}"
)
chain_translation = LLMChain(
    llm=llm, prompt=prompt_translation, output_key="german_translation"
)

overall_chain = SequentialChain(
    chains=[chain_review, chain_comment, chain_summary, chain_translation],
    input_variables=["dish_name", "experience"],
    output_variables=["review", "comment", "summary", "german_translation"],
)

overall_chain({"dish_name": "Pizza Salami", "experience": "It was awful!"})
