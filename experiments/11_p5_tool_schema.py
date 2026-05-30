"""
P5 도구 인수 추론 실패 심층 분석
- filesystem read_text_file 실제 스키마 확인
- head/tail 파라미터 단독 사용 테스트
- 우회 방법 탐색
"""

import json, time, torch
from pathlib import Path
from mcp import StdioServerParameters, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient
import asyncio

LOG_DIR = Path(__file__).parent.parent / "logs"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SERVER_PARAMS = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem",
          str(Path(__file__).parent.parent)],
)


async def get_tool_schema():
    """MCP 서버에서 read_text_file 실제 스키마 조회"""
    async with stdio_client(SERVER_PARAMS) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = await session.list_tools()
            for tool in tools.tools:
                if "read" in tool.name.lower() or "file" in tool.name.lower():
                    print(f"\n  도구명: {tool.name}")
                    print(f"  설명: {tool.description}")
                    schema = tool.inputSchema
                    print(f"  스키마:\n{json.dumps(schema, indent=4, ensure_ascii=False)}")


async def test_direct_calls():
    """파라미터 조합별 직접 호출 테스트"""
    test_cases = [
        {"path": "README.md"},
        {"path": "README.md", "tail": 5},
        {"path": "README.md", "head": 5},
        {"path": "README.md", "head": 5, "tail": 5},
    ]
    results = []
    async with stdio_client(SERVER_PARAMS) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            for args in test_cases:
                try:
                    result = await session.call_tool("read_text_file", arguments=args)
                    content = str(result.content[0].text if result.content else "")[:200]
                    results.append({"args": args, "ok": True, "content": content})
                    print(f"  ✅ {args} → {content[:80]}")
                except Exception as e:
                    results.append({"args": args, "ok": False, "error": str(e)})
                    print(f"  ❌ {args} → {e}")
    return results


def run_agent_with_hint(query, model, tool_hint):
    """도구 사용 힌트를 포함한 쿼리로 에이전트 실행"""
    t0 = time.time()
    try:
        with MCPClient(SERVER_PARAMS) as tools:
            agent = ToolCallingAgent(tools=tools, model=model, max_steps=4)
            result = agent.run(f"{query}\n\n힌트: {tool_hint}")
            return True, str(result)[:200], round(time.time()-t0, 1)
    except Exception as e:
        return False, str(e)[:200], round(time.time()-t0, 1)


def main():
    print(f"\n{'='*65}")
    print("  실험 A: read_text_file 실제 스키마 확인")
    print(f"{'='*65}")
    asyncio.run(get_tool_schema())

    print(f"\n{'='*65}")
    print("  실험 B: 파라미터 조합별 직접 호출")
    print(f"{'='*65}")
    schema_results = asyncio.run(test_direct_calls())

    print(f"\n{'='*65}")
    print("  실험 C: 에이전트에 도구 힌트 제공")
    print(f"{'='*65}")
    model = TransformersModel(MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16)

    hint_cases = [
        ("힌트없음",    "README.md 파일 내용을 한국어로 요약해줘.",
         ""),
        ("tail힌트",   "README.md 파일 내용을 한국어로 요약해줘.",
         "read_text_file을 사용할 때 tail 파라미터에 숫자(예: 50)를 반드시 넣어야 해."),
        ("list후요약", "README.md 파일 내용을 한국어로 요약해줘.",
         "파일을 직접 읽는 대신 list_directory로 파일 구조만 파악하고 알려줘."),
        ("대안도구",   "README.md 파일 내용을 한국어로 요약해줘.",
         "read_file이나 get_file_contents 같은 다른 도구를 사용해봐."),
    ]

    hint_results = []
    for label, query, hint in hint_cases:
        if hint:
            ok, resp, elapsed = run_agent_with_hint(query, model, hint)
        else:
            try:
                with MCPClient(SERVER_PARAMS) as tools:
                    agent = ToolCallingAgent(tools=tools, model=model, max_steps=4)
                    t0 = time.time()
                    result = agent.run(query)
                    ok, resp, elapsed = True, str(result)[:200], round(time.time()-t0, 1)
            except Exception as e:
                ok, resp, elapsed = False, str(e)[:100], 0.0

        icon = "✅" if ok else "❌"
        print(f"  {icon} [{label:<10}] {elapsed:>5}s | {resp[:60]}")
        hint_results.append({"label": label, "ok": ok, "elapsed": elapsed, "resp": resp})

    # ── 분석 요약 ──
    print(f"\n{'#'*65}")
    print("  P5 분석 요약")
    print(f"{'#'*65}")

    print("\n  직접 호출 결과:")
    for r in schema_results:
        icon = "✅" if r["ok"] else "❌"
        print(f"    {icon} {r['args']}")
        if not r["ok"]:
            print(f"       오류: {r.get('error', '')[:80]}")
        else:
            print(f"       내용: {r.get('content', '')[:60]}")

    print("\n  힌트 제공 효과:")
    for r in hint_results:
        icon = "✅" if r["ok"] else "❌"
        print(f"    {icon} [{r['label']:<10}] → {r['resp'][:60]}")

    # 저장
    log = LOG_DIR / "p5_tool_schema.json"
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"schema_results": schema_results, "hint_results": hint_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  로그: {log}")


if __name__ == "__main__":
    main()
