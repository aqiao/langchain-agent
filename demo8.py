from botocore.exceptions import ClientError
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import boto3

# Get the prompt to use - you can modify this!
# there is access limitation 100 for below url, once exceeded will failed
# from langchain import hub
# prompt = hub.pull("hwchase17/openai-tools-agent")
# so it's recommended to store the prompt to local
system_prompt = """SYSTEM

You are a helpful assistant

PLACEHOLDER

chat_history
HUMAN

{input}

PLACEHOLDER
agent_scratchpad
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")]
)
model = ChatOpenAI(model="gpt-3.5-turbo-1106")
s3_client = boto3.client("s3")
ses_client = boto3.client("ses")


@tool
def create_s3_bucket(bucket_name: str) -> str:
    """
    Create s3 bucket
    """
    try:
        response = s3_client.list_buckets()
        # bucket = filter(lambda item: bucket_name == item['Name'], response['Buckets'])
        # print(len(bucket))
        # if bucket:
        #     return f"{bucket_name} is existed,create bucket failed"
        s3_client.create_bucket(Bucket=bucket_name)
        return f"bucket {bucket_name} creation succeeded"
    except ClientError as error:
        return f"bucket {bucket_name} creation failed"


@tool
def send_email(to: str, content: str) -> str:
    """set create_s3_bucket function value as content"""
    # send_args = {
    #     "Source": "vke@live.com",
    #     "Destination": {'ToAddresses': [to]},
    #     "Message": {
    #         "Subject": {"Data": "Langchain Agent 邮件测试"},
    #         "Body": {"Text": {"Data": create_result}, "Html": {"Data": create_result}},
    #     },
    # }
    # try:
    #     ses_client.send_email(**send_args)
    #     return "send succeeded"
    # except ClientError as error:
    #     return str(error)
    return f"send {content} to {to}"


tools = [create_s3_bucket, send_email]
if __name__ == '__main__':
    
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, returnIntermediateSteps=True)
    # for chunk in agent_executor.stream({
    #         "input": "创建一个名称为langchain-agent-demo-bucket的桶，将操作结果发送给jianqiaohe@nwcdcloud.cn"
    #     }):
    #     # Agent Action
    #     if "actions" in chunk:
    #         for action in chunk["actions"]:
    #             print(
    #                 f"Calling Tool ```{action.tool}``` with input ```{action.tool_input}```"
    #             )
    #     # Observation
    #     elif "steps" in chunk:
    #         for step in chunk["steps"]:
    #             print(f"Got result: ```{step.observation}```")
    agent_executor.invoke(
        {
             "input": "创建名字为langchain-agent-demo-bucket-liuyingyu的桶，并将创建结果发送给jianqiaohe@nwcdcloud.cn"
        }
    #     # {
    #     #     "input": "create a s3 bucket named langchain-agent-demo-bucket, if the return value of function create_s3_bucket contains successful, please send email to me"
    #     # }
    )
    # create_s3_bucket('langchain-agent-demo-bucket')
