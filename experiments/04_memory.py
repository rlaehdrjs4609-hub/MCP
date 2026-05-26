"""
MCP 실험 #4 - memory
지식 그래프 기반 장기 메모리: 한국어 개념 저장 및 검색
"""

from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

mcp_server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-memory"],
)

QUERIES = [
    "다음 한국어 AI 용어들을 메모리에 저장해줘: MCP는 'Model Context Protocol'의 약자로 AI 모델과 외부 도구를 연결하는 프로토콜이야. RAG는 '검색 증강 생성'으로 외부 문서를 검색해서 답변을 생성하는 방식이야.",
    "아까 저장한 MCP와 RAG에 대한 정보를 불러와서 두 기술의 차이점을 한국어로 설명해줘",
    "새로운 개념을 추가해줘: LLM은 '대형 언어 모델'로 GPT, Claude, Qwen 같은 모델들이 해당돼. MCP, RAG, LLM 세 개념의 관계를 메모리에서 찾아서 한국어로 설명해줘",
]

def run():
    model = TransformersModel(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto",
        torch_dtype="auto",
    )

    with MCPClient(mcp_server_params) as tools:
        agent = ToolCallingAgent(tools=tools, model=model, max_steps=4)

        for query in QUERIES:
            print(f"\n{'='*60}\n쿼리: {query}\n{'='*60}")
            result = agent.run(query)
            print(f"결과:\n{result}")

if __name__ == "__main__":
    run()
