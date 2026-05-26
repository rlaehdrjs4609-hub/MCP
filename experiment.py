"""
로컬 HuggingFace 모델 + MCP 웹검색 서버 한국어 쿼리 실험
"""

import os
from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

load_dotenv()

# --- 모델 설정 ---
# Qwen2.5-3B: 캐시된 모델 중 한국어 성능 최우수
# CPU 환경이므로 짧은 쿼리 권장 (응답에 수 분 소요될 수 있음)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# --- MCP 서버 선택 ---
# Brave 또는 Tavily 중 API 키가 있는 것을 사용
brave_key = os.getenv("BRAVE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if brave_key:
    mcp_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": brave_key},
    )
    print("Brave Search MCP 서버 사용")
elif tavily_key:
    mcp_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp"],
        env={"TAVILY_API_KEY": tavily_key},
    )
    print("Tavily MCP 서버 사용")
else:
    raise ValueError(
        ".env 파일에 BRAVE_API_KEY 또는 TAVILY_API_KEY를 설정해주세요.\n"
        ".env.example 파일을 참고하세요."
    )

# --- 한국어 실험 쿼리 ---
KOREAN_QUERIES = [
    "2025년 한국 AI 스타트업 투자 현황을 검색해줘",
    "MCP(Model Context Protocol)란 무엇인지 최신 정보로 설명해줘",
    "한국어 LLM 벤치마크 최신 결과를 찾아줘",
]

# --- 실행 ---
def run_experiment(query: str):
    print(f"\n{'='*60}")
    print(f"쿼리: {query}")
    print('='*60)

    model = TransformersModel(
        model_id=MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )

    with MCPClient(mcp_server_params) as mcp_tools:
        agent = ToolCallingAgent(
            tools=mcp_tools,
            model=model,
            max_steps=3,
        )
        result = agent.run(query)
        print(f"\n[결과]\n{result}")
    return result


if __name__ == "__main__":
    # 첫 번째 쿼리만 실행 (테스트용)
    run_experiment(KOREAN_QUERIES[0])

    # 전체 실험 시 아래 주석 해제
    # for query in KOREAN_QUERIES:
    #     run_experiment(query)
