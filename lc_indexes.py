import os
import pinecone

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
from langchain.chains import RetrievalQA

load_dotenv(find_dotenv())
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

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
    environment=os.environ["PINECONE_ENV"],  # next to api key in console
)

index_name = "langchain-tut"

vector_store = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
# vector_store = Pinecone.from_existing_index(index_name, embeddings)

query = "When does the restaurant open?"
docs = vector_store.similarity_search(query)

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

chain_type_kwargs = {"prompt": PROMPT}

chat = RetrievalQA.from_chain_type(llm=llmChat, chain_type="stuff",
                                   retriever=vector_store.as_retriever(), chain_type_kwargs=chain_type_kwargs,)

query_chat = "When does the restaurant open?"
out = chat.run(query_chat)

print(out)

query_chat2 = "What gluten free options do you have?"
out2 = chat.run(query_chat2)

print(out2)
