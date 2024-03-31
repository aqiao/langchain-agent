import coloredlogs
import logging
import argparse
from utils import opensearch, secret
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import tool
import boto3

# Setup bedrock
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)
# opensearch_client=boto3.client()

@tool
def say_hi(name: str) -> str:
    """Say hi to name"""
    return f"hi {name}"

def build_aoss_kb(bedrock_client, kb_name):
    """
    build a Amazon Bedrock knowledge base with Amazon OpenSearch Serverless
    :param bedrock_client:
    :param kb_name:
    :return:
    """
    pass

def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id,
        client=bedrock_client,
        model_kwargs={'temperature': 0},
        )
    return bedrock_llm

tools = load_tools(
    ["awslambda"],
    awslambda_tool_name="email_sender",
    awslambda_tool_description="sends an email with the specified content to test@testing123.com",
    function_name="send_email",
)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    llm = create_bedrock_llm(bedrock_client, "amazon.titan-text-express-v1")
    agent = initialize_agent(
        [say_hi], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True
    )

    result = agent.run("say hi to Tom")
    print(result)
