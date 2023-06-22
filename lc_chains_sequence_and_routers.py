import os
import openai
import json
from dotenv import load_dotenv, find_dotenv
import json
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
from langchain.chains.router import MultiPromptChain

# router
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# This is an LLMChain to write a review given a dish name and the experience.
prompt_review = ChatPromptTemplate.from_template(
    template="You ordered {dish_name} and your experience was {experience}. Write a review: "
)
chain_review = LLMChain(llm=llm, prompt=prompt_review, output_key="review")

# This is an LLMChain to write a follow-up comment given the restaurant review.
prompt_comment = ChatPromptTemplate.from_template(
    template="Given the restaurant review: {review}, write a follow-up comment: "
)
chain_comment = LLMChain(llm=llm, prompt=prompt_comment, output_key="comment")

# This is an LLMChain to summarize a review.
prompt_summary = ChatPromptTemplate.from_template(
    template="Summarise the review in one short sentence: \n\n {review}"
)
chain_summary = LLMChain(llm=llm, prompt=prompt_summary, output_key="summary")

# This is an LLMChain to translate a summary into German.
prompt_translation = ChatPromptTemplate.from_template(
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

# out = overall_chain({"dish_name": "Pizza Salami",
#                     "experience": "It was awful!"})

# print(out)

# print(json.dumps(out, indent=4))

# {
#     "dish_name": "Pizza Salami",
#     "experience": "It was awful!",
#     "review": "I recently ordered a Pizza Salami from a well-known restaurant and unfortunately, my experience was simply awful. From the moment I took a bite of the pizza, I could tell that something was off. The crust was dry and lacked any real flavor, and the sauce on top was so bland that I couldn't even taste it.\n\nBut it was the salami that really put me off. The slices were so thin and dry that they crumbled when I tried to bite into them. And the flavor was just not what I was expecting - it tasted like it had been sitting in a fridge for far too long.\n\nOverall, I was extremely disappointed with my Pizza Salami experience. I had high hopes for this dish, but it just didn't deliver. I would not recommend this dish to anyone, and I certainly won't be ordering it again.",
#     "comment": "After posting my review, I received a response from the restaurant apologizing for my negative experience with the Pizza Salami. They offered to provide a replacement dish or a refund for my order.\n\nI appreciate their prompt response and willingness to make things right. However, I decided not to take them up on their offer as I have lost trust in their ability to deliver a quality dish. I hope they take this feedback into consideration and work on improving their pizza recipe in the future.\n\nOverall, I commend the restaurant for their customer service, but unfortunately, I cannot recommend the Pizza Salami based on my disappointing experience.",
#     "summary": "The reviewer had an awful experience with a Pizza Salami from a well-known restaurant, with dry and flavorless crust and bland sauce, as well as thin and dry salami slices with a bad taste.",
#     "german_translation": "Der Rezensent hatte eine schreckliche Erfahrung mit einer Pizza Salami aus einem bekannten Restaurant, mit trockenem und geschmacklosem Teig und fade So\u00dfe, sowie d\u00fcnnen und trockenen Salamischeiben mit schlechtem Geschmack."
# }


# 2 :

llmNorm = OpenAI(model_name="text-davinci-001")

positive_template = """You are an AI that focuses on the positive side of things. \
Whenever you analyze a text, you look for the positive aspects and highlight them. \
Here is the text:
{input}"""

neutral_template = """You are an AI that has a neutral perspective. You just provide a balanced analysis of the text, \
not favoring any positive or negative aspects. Here is the text:
{input}"""

negative_template = """You are an AI that is designed to find the negative aspects in a text. \
You analyze a text and show the potential downsides. Here is the text:
{input}"""

prompt_infos = [
    {
        "name": "positive",
        "description": "Good for analyzing positive sentiments",
        "prompt_template": positive_template,
    },
    {
        "name": "neutral",
        "description": "Good for analyzing neutral sentiments",
        "prompt_template": neutral_template,
    },
    {
        "name": "negative",
        "description": "Good for analyzing negative sentiments",
        "prompt_template": negative_template,
    },
]


destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template)
    chain = LLMChain(llm=llmNorm, prompt=prompt)
    destination_chains[name] = chain
destination_chains


destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llmNorm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutral"],
    verbose=True,
)

routerOut = chain.run("I ordered Pizza Salami for 9.99$ and it was awesome!")

print(json.dumps(routerOut, indent=4))
