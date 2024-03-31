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
        s3_client.create_bucket(Bucket=bucket_name)
        return bucket_name
    except ClientError as error:
        return f"Creation failed since {str(error)}"


@tool
def send_email(to: str, content) -> str:
    """Send email with create bucket action result """
    send_args = {
        "Source": "vke@live.com",
        "Destination": {'ToAddresses': [to]},
        "Message": {
            "Subject": {"Data": "Langchain Agent 邮件测试"},
            "Body": {"Text": {"Data": content}, "Html": {"Data": content}},
        },
    }
    try:
        ses_client.send_email(**send_args)
        return "send succeeded"
    except ClientError as error:
        return str(error)


tools = [create_s3_bucket, send_email]
if __name__ == '__main__':
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # result = agent_executor.invoke(
    #     {
    #         "input": "创建一个s3桶，桶的名字是langchain-agent-demo-bucket。将创建结果作为邮件内容发送给jianqiaohe@nwcdcloud.cn"
    #     }
    # )
    result = agent_executor.invoke(
        {
            "input": "创建一个名称为langchain-agent-demo-bucket的桶，然后将创建结果发送给jianqiaohe@nwcdcloud.cn"
        }
    )
    # result = agent_executor.invoke(
    #     {
    #         "input": "create a s3 bucket named langchain-agent-demo-bucket, then send the creation result to "
    #                  "jianqiaohe@nwcdcloud.cn "
    #     }
    # )
    # result = send_email("jianqiaohe@nwcdcloud.cn","sending email with langchain agent demo")
    print(result)
    # result = create_s3_bucket('langchain-agent-demo-bucket')
    # print(result)
