from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

system_prompt = """SYSTEM

You are a helpful assistant

PLACEHOLDER

chat_history
HUMAN

{input}

PLACEHOLDER
{agent_scratchpad}
"""


prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad")]
)
model = ChatOpenAI(model="gpt-3.5-turbo-1106")


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

# @tool
# def add(a: int, b: int) -> int:
#     "Add two integers."
#     return a + b


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, add, exponentiate]
if __name__ == '__main__':
    agent = create_openai_tools_agent(model,tools,prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # result = agent_executor.invoke(
    #     {
    #         "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    #     }
    # )

    result = agent_executor.invoke(
        {
            "input": "3乘以4再加上8"
        }
    )
    # result = agent_executor.invoke(
    #     {
    #         "input": "2的平方乘以5再加上8"
    #     }
    # )
    # result = agent_executor.invoke(
    #     {
    #         "input": "重复运行5次3乘以4"
    #     }
    # )
    # result = agent_executor.invoke(
    #     {
    #         "input": "如果今天是周三，那么计算4乘以5，否则计算5的平方，请输出正确结果"
    #     }
    # )
    # result = agent_executor.invoke(
    #     {
    #         "input": "今天是周三吗，如果是请计算4乘以5，否则计算5的平方，请输出最终结果"
    #     }
    # )