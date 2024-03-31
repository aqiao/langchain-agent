from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.llms.bedrock import Bedrock
import boto3
from langchain_core.prompts import PromptTemplate
from langchain import hub

# reference https://github.com/langchain-ai/langchain/issues/12620
react_prompt_template="""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {inputText}
Thought:{agent_scratchpad}
"""
# prompt = hub.pull("hwchase17/react")
prompt = PromptTemplate(
    input_variables=["inputText"],
    template=react_prompt_template
)

@tool
def say_hi(name: str) -> str:
    """Say hi to name"""
    return f"hi {name}"


def specify_bedrock_titan_llm():
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )
    # https://github.com/langchain-ai/langchain/issues/16840
    bedrock_llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock_client,
        model_kwargs={'textGenerationConfig': {"stopSequences": "Observation",'temperature': 0}}
    )
    return bedrock_llm


if __name__ == '__main__':
    llm = specify_bedrock_titan_llm()
    agent = create_react_agent(llm, [say_hi], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[say_hi], verbose=True, handle_parsing_errors=True)
    result = agent_executor.invoke({"inputText": "User: say hi to Tom\nBot:"})
    print(result)