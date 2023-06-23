import os
import pinecone
import json
import pprint
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
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryBufferMemory

# Documents and text
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vectors
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

# agents and tools
from langchain.agents import AgentType, load_tools, initialize_agent
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun


load_dotenv(find_dotenv())

# settings ===
index_name = "resturant"
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"],
)
# ===

if index_name not in pinecone.list_indexes():
    print('Please wait initilizing vector store')
    pinecone.create_index(
        index_name,
        dimension=1536
    )

    print('Loading vector store')
    loader = DirectoryLoader(
        "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vector_store = Pinecone.from_documents(
        docs, embeddings, index_name=index_name)

    print('Vector store initialized')
else:
    # if you already have an index, you can load it like this
    embeddings = OpenAIEmbeddings()
    vector_store = Pinecone.from_existing_index(index_name, embeddings)


# query = "When does the restaurant open?"
# docs = vector_store.similarity_search(query)
# print(docs[0].page_content)

# llm = OpenAI(temperature=0)
llmChat = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt_template = """You are a helpful assistant for our restaurant.

{context}

Question: {question}
Answer here:"""

PROMPT = ChatPromptTemplate.from_template(
    template=prompt_template
)


tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llmChat)
tools

#  or
# from langchain.agents import Tool
# tool_list = [
#     Tool(
#         name = "Math Tool",
#         func=tools[0].run,
#         description="Tool to calculate, nothing else"
#     )
# ]


# agent = initialize_agent(tools,
#                          llmChat,
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                          verbose=True)
# agent.run("How are you?")
# agent.run("What is 100 devided by 25?")


class CustomSearchTool(BaseTool):
    name = "restaurant search"
    description = "useful for when you need to answer questions about our restaurant"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = vector_store.as_retriever(search_type="mmr")
        docs = store.get_relevant_documents(query)
        print(docs)
        print("===========================")
        text_list = [doc.page_content for doc in docs]
        # print(text_list)
        return "\n".join(text_list)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


tools = [CustomSearchTool()]
agent = initialize_agent(
    tools, llmChat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("When does the restaurant open?")
agent.run("Do you have American cuisine?")
agent.run("Is there a happy hour")

# chat needs work
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True)
# agent = initialize_agent(
#     tools, llmChat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)


# agent.run(input="When does the restaurant open?")
