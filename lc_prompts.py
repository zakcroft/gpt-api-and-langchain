import os
import openai
import json

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]


# 1: basic
# output = llm("Tell me a joke")
# print('1', output)
# print(repr(output))  # with new lines


chat = ChatOpenAI(temperature=0)

# 2: Schema methods
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]

# outtwo = chat(messages)
# print('2', messages)
# print('2', outtwo)

# 3 from templates
template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# get a chat completion from the formatted messages
outthree = chat(
    chat_prompt.format_prompt(
        input_language="English", output_language="French", text="I love programming."
    ).to_messages()
)

# outthree = chat(messages)
# print('3', chat_prompt)
# print('3', outthree)

# 4 Chat template chain method
template = """
    Interprete the text and evaluate the text.
    sentiment: is the text in a positive, neutral or negative sentiment?
    subject: What subject is the text about? Use exactly one word.

    Format the output as JSON with the following keys:
    sentiment
    subject

    text: {input}
"""

llm = OpenAI(model_name="text-davinci-003")
llmChat = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt_template = ChatPromptTemplate.from_template(template=template)
chain = LLMChain(llm=llmChat, prompt=prompt_template)
output = chain.predict(input="I ordered Pizza Salami and it was awesome!")
# json_object = json.loads(prompt_template)

# json_formatted_str = json.dumps(json_object, indent=2)
# print('4', prompt_template)

# print('4', chain)
print('4', output)
