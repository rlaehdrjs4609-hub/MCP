"""
실험 #8 — 수정 효과 검증
1) structured_output=False  vs  True  (전체 서버)
2) 시스템 프롬프트 간결성 지시 효과
"""

import os, json, time, torch
from pathlib import Path
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

LOG_DIR = Path(__file__).parent.parent / "logs"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# 이전 실험에서 실패했던 한국어 쿼리들만 사용
FAILING_QUERIES = {
    "filesystem": "README.md 파일을 읽고 내용을 한국어로 요약해줘.",
    "github":     "rlaehdrjs4609-hub/MCP 저장소의 최근 커밋 3개를 한국어로 요약해줘.",
    "memory":     "다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야.",
}

SERVER_PARAMS = {
    "filesystem": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem",
              str(Path(__file__).parent.parent)],
    ),
    "github": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
    ),
    "memory": StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
    ),
}

# 시스템 프롬프트 변형 (memory 서버만)
SYSTEM_PROMPTS = {
    "none":    None,
    "concise": "You are a helpful assistant. When calling tools, keep all observations and string values concise (under 50 characters each). Always respond in Korean.",
}


def run_query(server_name, query, model, structured_output, system_prompt=None):
    t0 = time.time()
    try:
        with MCPClient(SERVER_PARAMS[server_name],
                       structured_output=structured_output) as tools:
            kwargs = {}
            if system_prompt:
                from smolagents.prompts import PromptTemplates
                kwargs["prompt_templates"] = PromptTemplates(
                    system_prompt=system_prompt
                )
            agent = ToolCallingAgent(tools=tools, model=model, max_steps=4, **kwargs)
            result = agent.run(query)
            return True, str(result)[:300], round(time.time()-t0, 1)
    except Exception as e:
        return False, str(e)[:200], round(time.time()-t0, 1)


def main():
    model = TransformersModel(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    results = []

    # ── 실험 A: structured_output False vs True ──
    print("\n" + "="*65)
    print("  실험 A: structured_output=False vs True (이전 실패 쿼리)")
    print("="*65)
    print(f"{'서버':<12} {'so=False':>10} {'so=True':>10}  비교")
    print("-"*65)

    for server, query in FAILING_QUERIES.items():
        row = {"server": server, "query": query}
        for so in [False, True]:
            ok, resp, elapsed = run_query(server, query, model, so)
            row[f"so_{so}"] = {"ok": ok, "elapsed": elapsed, "resp": resp}
            status = f"{'✅' if ok else '❌'} {elapsed}s"
        results.append(row)
        print(f"{server:<12} "
              f"{'✅' if row['so_False']['ok'] else '❌'} {row['so_False']['elapsed']:>4}s  "
              f"{'✅' if row['so_True']['ok'] else '❌'} {row['so_True']['elapsed']:>4}s  "
              f"{'개선됨' if row['so_False']['ok'] and not row['so_True']['ok'] else '동일' if row['so_False']['ok'] == row['so_True']['ok'] else '악화'}")

    # ── 실험 B: 시스템 프롬프트 간결성 지시 (memory 서버) ──
    print("\n" + "="*65)
    print("  실험 B: 시스템 프롬프트 간결성 지시 효과 (memory 서버)")
    print("="*65)
    query = FAILING_QUERIES["memory"]
    prompt_results = {}
    for name, sp in SYSTEM_PROMPTS.items():
        for so in [False, True]:
            key = f"sp={name}_so={so}"
            ok, resp, elapsed = run_query("memory", query, model,
                                          structured_output=so,
                                          system_prompt=sp)
            prompt_results[key] = {"ok": ok, "elapsed": elapsed, "resp": resp[:150]}
            icon = "✅" if ok else "❌"
            print(f"  {key:<30} {icon} {elapsed}s")
            if ok:
                print(f"    응답: {resp[:100]}")

    # ── 최종 요약 ──
    print("\n" + "#"*65)
    print("  수정 효과 요약")
    print("#"*65)
    for r in results:
        s = r["server"]
        f = r["so_False"]["ok"]; t = r["so_True"]["ok"]
        if f and not t:
            verdict = "✅ structured_output=False 로 수정됨"
        elif f and t:
            verdict = "✅ 둘 다 성공"
        elif not f and not t:
            verdict = "❌ 둘 다 실패 (다른 원인)"
        else:
            verdict = "⚠️ True만 성공 (예상 외)"
        print(f"  [{s}] {verdict}")

    # 로그 저장
    log = LOG_DIR / "fix_verification.json"
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"results": results, "prompt_results": prompt_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  로그 저장: {log}")


if __name__ == "__main__":
    main()
