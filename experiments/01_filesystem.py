"""
MCP 실험 #1 - filesystem
로컬 파일 읽기/쓰기를 모델이 직접 수행
"""

from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

TARGET_DIR = "/home/kdg4609/LLM/MCP"

mcp_server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", TARGET_DIR],
)

QUERIES = [
    "현재 디렉토리에 어떤 파일들이 있는지 목록을 알려줘",
    "README.md 파일 내용을 읽고 한국어로 요약해줘",
    "실험 결과를 저장할 'results.txt' 파일을 만들고 '실험 시작: filesystem MCP 테스트 완료' 라고 써줘",
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
