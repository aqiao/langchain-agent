from langchain.llms.bedrock import Bedrock
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import tool, StructuredTool,Tool
import boto3

# PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
# FORMAT_INSTRUCTIONS = """Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question"""
# SUFFIX = """Begin!
#
# Question: {input}
# Thought:{agent_scratchpad}"""

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: extract user name from question
Action: the action to take, should be one of [{tool_names}]
Action Input: extracted user name from question
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def say_hi(alias: str) -> str:
    '''
    say hi to param
    :param alias:
    :return:a string
    '''
    return f"你好 {alias}"


def create_bedrock_llm(client, model_version_id):
    bedrock_llm = Bedrock(
        model_id=model_version_id,
        client=client,
        model_kwargs={'temperature': 0},
        )
    return bedrock_llm


if __name__ == '__main__':
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    tool = StructuredTool.from_function(say_hi)
    tools = [tool]
    # my_tool = Tool("say_hi",say_hi,"say hi to the function param")
    llm = create_bedrock_llm(bedrock_client, "amazon.titan-text-express-v1")
    agent = initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        handle_parsing_errors=True,
        max_iterations=120,
        stop=["\nObservation:"],
        # agent_kwargs={
        #     'prefix': PREFIX,
        #     'format_instructions': FORMAT_INSTRUCTIONS,
        #     'suffix': SUFFIX,
        #     "input_variables": ["input", "agent_scratchpad"]
        # }
    )
    result = agent.invoke({"input": "tom"})
    # result = tool.run("tom")
    print(result)
