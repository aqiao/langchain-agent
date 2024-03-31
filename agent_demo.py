from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.llms.bedrock import Bedrock
import boto3
from langchain_core.prompts import PromptTemplate
from langchain import hub


@tool
def search(num: int) -> str:
    """Look up things online."""
    if num > 10:
        return "yes"
    return "no"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def say_hi(name: str) -> str:
    """Say hi to the world"""
    return f"hi {name}"


def specify_bedrock_titan_llm():
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    bedrock_llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={'temperature': 0}
    )
    return bedrock_llm


if __name__ == '__main__':
    # print(multiply.name)
    # print(multiply.description)
    # print(multiply.args)
    template = '''User Prompt: 
    You are good at math calculation:

    {tools}
    Action: the action to take, should be one of [{tool_names}]
    Final Answer: the final answer to the original input question

    User: {input}
    # Thought:{agent_scratchpad}
    Bot:
    '''
    # prompt_template = PromptTemplate(
    #     input_variables=["input"],
    #     template=template
    # )
    # prompt = prompt_template.format(a=4, b=3)
    # print(prompt)
    llm = specify_bedrock_titan_llm()
    # prompt = PromptTemplate.from_template("User: 北京今天天气怎么样")
    # Verified with string directly
    # result = llm.invoke("北京今天天气怎么样")

    # Verified titan prompt template
#     prompt = """
# At a Halloween party, Jack gets 15 candies.
# Jack eats 5 candies. He wants to give each friend
# 5 candies. How many friends can receive candies?
#
# Think step-by-step to come up with the right answer.
#
#     """
    template = """The following is text from a restaurant review:
{Text}
Summarize the category in one sentence"""
    prompt = PromptTemplate(
        input_variables=["Text"],
        template=template
    )
    text_value = """
    “I finally got to check out Alessandro’s Brilliant Pizza 
and it is now one of my favorite restaurants in Seattle. 
The dining room has a beautiful view over the Puget Sound 
but it was surprisingly not crowed. I ordered the fried 
castelvetrano olives, a spicy Neapolitan-style pizza 
and a gnocchi dish. The olives were absolutely decadent, 
and the pizza came with a smoked mozzarella, which was delicious. 
The gnocchi was fresh and wonderful. The waitstaff were attentive, 
and overall the experience was lovely. I hope to return soon.”
    """
    # formatted_value = template.format(Text=text_value)
    # print(formatted_value)
    # result = llm.invoke(formatted_value)
    agent = create_react_agent(llm, [say_hi], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True, handle_parsing_errors=True)
    result = agent_executor.invoke({"Text": text_value})
    print(result)
