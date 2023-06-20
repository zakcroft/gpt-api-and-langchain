import os
import json
import openai
import wikipedia
from dotenv import load_dotenv
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def chat(input):
    messages = [
    {"role": "system", "content": "You are a super organised personal assistant who uses minimal words and is very succinct. If the location is asked for always give back the populations and the temperture"},
    {"role": "user", "content": input}
    ]

    functions = [
        {
            "name": "get_info_from_wiki",
            "description": "Get further info from wiki for location and the population",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "population": {"type": "number"},
                },
                "required": ["location", "population"],
            },
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=0,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response["choices"][0]["message"]
    print('response_message =======', response)

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_info_from_wiki": get_info_from_wiki,
        }  # only one function in this example, but you can have multiple

        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
            population=function_args.get("population"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response

        print("=second_response=", second_response)
        return second_response
    else:
        print("=NO FUNCTION CALLING=")
        return response.choices[0].message["content"]

def get_info_from_wiki(location, population, unit="fahrenheit"):
    print("=get_info_from_wiki=")

    wiki = WikipediaAPIWrapper(doc_content_chars_max=5000, load_all_available_meta=True)
    wiki_research = wiki.run("Get further info on this location. Give back the time" + location)

    print('=wiki response=', wiki_research)
    
    return json.dumps(wiki_research)


output = chat("Tell me about france")
output2 = chat("Tell me about ice")
print('===', output)
print('===', output2)
