"""
MCP 실험 #2 - fetch
URL을 주면 모델이 직접 웹페이지 내용을 가져와서 처리
"""

from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

mcp_server_params = StdioServerParameters(
    command="npx",
    args=["-y", "mcp-fetch"],
)

QUERIES = [
    "https://en.wikipedia.org/wiki/Model_Context_Protocol 페이지를 가져와서 MCP가 뭔지 한국어로 설명해줘",
    "https://github.com/modelcontextprotocol/servers 페이지를 가져와서 어떤 공식 MCP 서버들이 있는지 한국어로 정리해줘",
]

def run():
    model = TransformersModel(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        torch_dtype="auto",
    )

    with MCPClient(mcp_server_params) as tools:
        agent = ToolCallingAgent(tools=tools, model=model, max_steps=3)

        for query in QUERIES:
            print(f"\n{'='*60}\n쿼리: {query}\n{'='*60}")
            result = agent.run(query)
            print(f"결과:\n{result}")

if __name__ == "__main__":
    run()
