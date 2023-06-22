from langchain.memory import ChatMessageHistory
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
from langchain.chains import ConversationChain

# Memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

load_dotenv(find_dotenv())

# openai.api_key = os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)
llmChat = ChatOpenAI(model_name="gpt-3.5-turbo")


# history = ChatMessageHistory()

# history.add_user_message("hi!")
# history.add_ai_message("hello my friend!")
# print(history.messages)

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("hello my friend!")


# conversation = ConversationChain(
#     llm=llmChat, verbose=True, memory=memory
# )
# conversation.run(input="Hi")
# out = conversation.run(input="I need to know the capital of france")

# print(out)

# out2 = conversation.run(input="How far is it from london")
# print(out2)

# print(memory.load_memory_variables({}))


# cheaper way with summaries
# This is better as it summarizes and adds to the system prompt
review = "I ordered Pizza Salami for 9.99$ and it was awesome! \
The pizza was delivered on time and was still hot when I received it. \
The crust was thin and crispy, and the toppings were fresh and flavorful. \
The Salami was well-cooked and complemented the cheese perfectly. \
The price was reasonable and I believe I got my money's worth. \
Overall, I am very satisfied with my order and I would recommend this pizza place to others."

memory = ConversationSummaryBufferMemory(llm=llmChat, max_token_limit=100)

memory.save_context(
    {"input": "Could you analyze a review for me?"},
    {"output": "Sure, I'd be happy to. Could you provide the review?"},
)
# memory.save_context(
#     {"input": f"{review}"},
#     {"output": "Thanks for the review"},
# )

conversation = ConversationChain(llm=llmChat, memory=memory, verbose=True)
conversation.predict(input="Thank you very much!")
# print(memory.load_memory_variables({}))
