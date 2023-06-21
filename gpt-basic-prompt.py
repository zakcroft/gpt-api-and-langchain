import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def chat(input):
    messages = [
        {"role": "system", "content": "You are a sarcastic assistant and likes a joke"},
        {"role": "user", "content": input}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


output = chat("What is the capital of france")
# Oh, I don't know, maybe it's Timbuktu? Just kidding, it's Paris, obviously.
print(output)
