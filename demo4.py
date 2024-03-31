from langchain.tools.render import render_text_description
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms.bedrock import Bedrock
import boto3

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together"""
    return first_int * second_int


def create_bedrock_llm(client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id,
        client=client,
        model_kwargs={'temperature': 0},
        )
    return bedrock_llm


def get_titan_llm():
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    return create_bedrock_llm(bedrock_client, "amazon.titan-text-express-v1")


if __name__ == '__main__':
    # print(multiply.name)
    # print(multiply.description)
    # print(multiply.args)
    # ret = multiply.invoke({"first_int":4, "second_int": 5})
    # print(ret)

    rendered_tools = render_text_description([multiply])
    system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON format with 'name' and 'arguments' keys.

"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    model = get_titan_llm()
    chain = prompt | model
    result = chain.invoke({"input": "what's thirteen times 4"})

    print(result)