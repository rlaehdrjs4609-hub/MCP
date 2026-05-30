"""
P1 언어 전환 문제 심층 분석
- 언제 한국어가 유지되고 언제 전환되는가?
- 시스템 프롬프트 효과
- 쿼리 유형별 패턴
"""

import os, json, time, torch
from pathlib import Path
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

LOG_DIR = Path(__file__).parent.parent / "logs"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SERVER = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem",
          str(Path(__file__).parent.parent)],
)

# ── 실험 A: 어떤 쿼리 유형에서 언어 전환이 발생하는가 ──
QUERY_TYPES = [
    ("단순_목록",    "현재 디렉토리에 있는 파일 목록을 알려줘."),
    ("파일_읽기",    "README.md 파일 내용을 요약해줘."),
    ("한국어_강조",  "한국어로만 대답해줘. 파일 목록을 알려줘."),
    ("영어혼합",     "Please list files. 한국어로 답해줘."),
    ("짧은",        "파일 목록?"),
    ("정중체",       "현재 디렉토리의 파일과 폴더 목록을 한국어로 상세히 알려주시겠어요?"),
]

# ── 실험 B: 시스템 프롬프트 변형 ──
SYSTEM_PROMPTS = {
    "없음": None,
    "한국어_지시": "당신은 한국어 전용 어시스턴트입니다. 모든 응답은 반드시 한국어로만 작성하세요. 절대 영어나 다른 언어를 사용하지 마세요.",
    "역할_부여":   "당신은 한국어 AI 연구자입니다. 사용자가 한국어로 질문하면 반드시 한국어로만 답하세요.",
}

QUERY_B = "현재 디렉토리에 있는 모든 파일과 폴더 목록을 알려줘."


def detect_language(text: str) -> str:
    if not text or len(text) < 5:
        return "unknown"
    ko = sum(1 for c in text if "가" <= c <= "힣")
    zh = sum(1 for c in text if "一" <= c <= "鿿")
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    total = max(ko + zh + en, 1)
    if ko / total > 0.3: return "한국어"
    if zh / total > 0.3: return "중국어"
    if en / total > 0.3: return "영어"
    return "혼합"


def run(query, model, system_prompt=None):
    t0 = time.time()
    try:
        with MCPClient(SERVER) as tools:
            if system_prompt:
                # smolagents system_prompt: 쿼리 앞에 붙이는 방식으로 주입
                full_query = f"[시스템 지시: {system_prompt}]\n\n{query}"
                agent = ToolCallingAgent(tools=tools, model=model, max_steps=3)
                result = agent.run(full_query)
            else:
                agent = ToolCallingAgent(tools=tools, model=model, max_steps=3)
                result = agent.run(query)
            lang = detect_language(str(result))
            return True, str(result)[:200], lang, round(time.time()-t0, 1)
    except Exception as e:
        return False, str(e)[:100], "error", round(time.time()-t0, 1)


def main():
    model = TransformersModel(MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16)

    results = {"query_types": [], "system_prompts": []}

    # ── 실험 A ──
    print(f"\n{'='*65}")
    print("  실험 A: 쿼리 유형별 언어 전환 패턴")
    print(f"{'='*65}")
    print(f"{'유형':<12} {'언어':>6} {'시간':>6}  응답 앞부분")
    print("-"*65)

    for label, query in QUERY_TYPES:
        ok, resp, lang, elapsed = run(query, model)
        icon = {"한국어": "✅", "중국어": "🔴", "영어": "🟠", "혼합": "🟡"}.get(lang, "❓")
        print(f"{label:<12} {icon}{lang:>4} {elapsed:>5}s  {resp[:40]}")
        results["query_types"].append(
            {"label": label, "query": query, "lang": lang, "ok": ok, "elapsed": elapsed, "resp": resp}
        )

    # ── 실험 B ──
    print(f"\n{'='*65}")
    print("  실험 B: 시스템 프롬프트로 언어 전환 억제 가능한가")
    print(f"{'='*65}")
    print(f"{'시스템프롬프트':<15} {'언어':>6} {'시간':>6}  응답 앞부분")
    print("-"*65)

    for sp_label, sp in SYSTEM_PROMPTS.items():
        ok, resp, lang, elapsed = run(QUERY_B, model, system_prompt=sp)
        icon = {"한국어": "✅", "중국어": "🔴", "영어": "🟠", "혼합": "🟡"}.get(lang, "❓")
        print(f"{sp_label:<15} {icon}{lang:>4} {elapsed:>5}s  {resp[:40]}")
        results["system_prompts"].append(
            {"sp_label": sp_label, "lang": lang, "ok": ok, "elapsed": elapsed, "resp": resp}
        )

    # ── 분석 요약 ──
    print(f"\n{'#'*65}")
    print("  P1 분석 요약")
    print(f"{'#'*65}")

    switch_cases = [r for r in results["query_types"] if r["lang"] != "한국어"]
    ko_cases     = [r for r in results["query_types"] if r["lang"] == "한국어"]
    print(f"\n  쿼리 유형별: 한국어 유지 {len(ko_cases)}/{len(QUERY_TYPES)}, 전환 {len(switch_cases)}/{len(QUERY_TYPES)}")
    for r in switch_cases:
        print(f"    - [{r['label']}] → {r['lang']}")

    sp_ko = [r for r in results["system_prompts"] if r["lang"] == "한국어"]
    print(f"\n  시스템 프롬프트: 한국어 유지 {len(sp_ko)}/{len(SYSTEM_PROMPTS)}")
    for r in results["system_prompts"]:
        icon = "✅" if r["lang"] == "한국어" else "❌"
        print(f"    {icon} [{r['sp_label']}] → {r['lang']}")

    log = LOG_DIR / "p1_language_switch.json"
    with open(log, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  로그: {log}")


if __name__ == "__main__":
    main()
