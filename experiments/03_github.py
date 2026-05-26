"""
MCP 실험 #3 - github
모델이 GitHub 저장소를 직접 조회/수정
"""

import os
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN 환경변수를 설정해주세요: export GITHUB_TOKEN=$(gh auth token)")
REPO_OWNER = "rlaehdrjs4609-hub"
REPO_NAME = "MCP"

mcp_server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
    env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
)

QUERIES = [
    f"{REPO_OWNER}/{REPO_NAME} 저장소의 최근 커밋 목록을 가져와서 한국어로 요약해줘",
    f"{REPO_OWNER}/{REPO_NAME} 저장소에 'MCP 실험 시작' 이라는 제목으로 이슈를 한국어로 작성해줘. 내용은 로컬 모델로 MCP 서버 연결 실험 중이라고 써줘",
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
