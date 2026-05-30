"""
실험 #9 — Qwen2.5-7B로 동일 실험 재현
목적: 3B에서 발견한 6가지 한국어 쿼리 문제가 7B에서도 재현되는지 확인

문제 목록 (3B 기준):
  P1. 언어 전환      — 오류 반복 시 한국어→영어 전환
  P2. 내용 번역·손실 — 한국어 관찰값을 영어로 변환 저장 (memory)
  P3. JSON 생성 오류 — 긴 JSON에서 outer } 누락 (memory)
  P4. 에이전트 오동작 — 잘못된 도구 호출·환각 (github)
  P5. 도구 인수 추론 실패 — tail 인수 반복 오류 (filesystem)
  P6. 컨텍스트 폭증 — 많은 API 호출로 토큰 폭증 (github)
"""

import os, json, time, torch
from pathlib import Path
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

LOG_DIR = Path(__file__).parent.parent / "logs"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# 3B 실험과 동일한 쿼리 쌍 (EN/KO)
QUERY_PAIRS = {
    "filesystem": [
        {"en": "List all files and folders in the current directory.",
         "ko": "현재 디렉토리에 있는 모든 파일과 폴더 목록을 알려줘."},
        {"en": "Read the README.md file and summarize its contents.",
         "ko": "README.md 파일을 읽고 내용을 한국어로 요약해줘."},  # P5 트리거 쿼리
    ],
    "github": [
        {"en": "List the files in the rlaehdrjs4609-hub/MCP repository.",
         "ko": "rlaehdrjs4609-hub/MCP 저장소의 파일 목록을 알려줘."},
        {"en": "Summarize the 3 most recent commits in rlaehdrjs4609-hub/MCP.",
         "ko": "rlaehdrjs4609-hub/MCP 저장소의 최근 커밋 3개를 한국어로 요약해줘."},  # P4 트리거
    ],
    "memory": [
        {"en": "Store: MCP is a protocol connecting AI models to external tools. Qwen is a multilingual LLM by Alibaba.",
         "ko": "다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야."},  # P2/P3 트리거
        {"en": "Retrieve MCP and Qwen info from memory and explain their relationship.",
         "ko": "메모리에서 MCP와 Qwen 정보를 불러와서 두 개의 관계를 한국어로 설명해줘."},
    ],
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


def detect_problems(lang, server, response, steps_log):
    """응답과 step 로그에서 3B에서 발견한 문제 패턴을 검사"""
    issues = []

    # P1: 언어 전환 (한국어 쿼리인데 영어 응답)
    if lang == "ko" and response:
        ko_ratio = sum(1 for c in response if "가" <= c <= "힣") / max(len(response), 1)
        if ko_ratio < 0.05 and len(response) > 30:
            issues.append("P1:언어전환")

    # P2: 한국어 내용 번역 (memory 서버, 저장된 내용이 영어인지)
    if lang == "ko" and server == "memory" and response:
        if any(kw in response.lower() for kw in ["protocol", "model", "alibaba", "language"]):
            if sum(1 for c in response if "가" <= c <= "힣") < 5:
                issues.append("P2:내용번역")

    # P3: JSON 생성 오류
    if any("Expecting ',' delimiter" in str(s) or "JSON blob" in str(s) for s in steps_log):
        issues.append("P3:JSON오류")

    # P4: 에이전트 오동작 (잘못된 저장소 생성 등)
    if any("create_repository" in str(s) for s in steps_log):
        issues.append("P4:오동작-저장소생성")
    if any("hallucin" in str(s).lower() or "simulation" in str(s).lower() for s in steps_log):
        issues.append("P4:오동작-환각")

    # P5: 도구 인수 반복 실패
    tail_errors = sum(1 for s in steps_log if "tail is required" in str(s))
    if tail_errors >= 2:
        issues.append(f"P5:인수추론실패({tail_errors}회)")

    # P6: 컨텍스트 폭증
    token_counts = [s.get("input_tokens", 0) for s in steps_log if isinstance(s, dict)]
    if token_counts and max(token_counts) > 30000:
        issues.append(f"P6:컨텍스트폭증({max(token_counts):,}토큰)")

    return issues


def run_query(server_name, lang, query, model):
    steps_log = []
    t0 = time.time()
    result = {"lang": lang, "query": query[:60], "ok": False,
              "response": None, "elapsed": None, "issues": [], "steps": 0}
    try:
        with MCPClient(SERVER_PARAMS[server_name]) as tools:
            agent = ToolCallingAgent(tools=tools, model=model, max_steps=5)

            # step 로그를 캡처하기 위해 run 실행
            response = agent.run(query)
            result["ok"] = True
            result["response"] = str(response)[:400]

            # agent의 memory에서 step 로그 추출
            for step in agent.memory.steps:
                log_entry = {"type": type(step).__name__}
                if hasattr(step, "model_output"):
                    log_entry["output"] = str(step.model_output)[:200]
                if hasattr(step, "observations"):
                    log_entry["obs"] = str(step.observations)[:200]
                if hasattr(step, "input_tokens"):
                    log_entry["input_tokens"] = step.input_tokens
                steps_log.append(log_entry)
            result["steps"] = len(steps_log)

    except Exception as e:
        result["response"] = str(e)[:200]
        steps_log.append(str(e))

    result["elapsed"] = round(time.time() - t0, 1)
    result["issues"] = detect_problems(lang, server_name, result["response"], steps_log)
    return result


def main():
    print(f"모델 로딩: {MODEL_ID}")
    model = TransformersModel(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
    )

    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for server, pairs in QUERY_PAIRS.items():
        if server == "github" and not GITHUB_TOKEN:
            print(f"[SKIP] github: GITHUB_TOKEN 미설정")
            continue

        print(f"\n{'='*65}")
        print(f"  [{server.upper()}] 7B 실험")
        print('='*65)
        server_results = []

        for pair in pairs:
            for lang in ["en", "ko"]:
                query = pair[lang]
                print(f"\n  [{lang.upper()}] {query[:55]}")
                r = run_query(server, lang, query, model)
                server_results.append(r)

                icon = "✅" if r["ok"] else "❌"
                issues_str = ", ".join(r["issues"]) if r["issues"] else "없음"
                print(f"  {icon} {r['elapsed']}s | 문제: {issues_str}")
                if r["response"]:
                    print(f"  응답: {r['response'][:120]}")

        all_results[server] = server_results

    # ── 3B vs 7B 비교 요약 ──
    print(f"\n\n{'#'*65}")
    print("  7B 실험 결과 — 문제 발생 현황")
    print(f"{'#'*65}")

    problem_counts = {}
    for server, entries in all_results.items():
        for e in entries:
            for issue in e["issues"]:
                pcode = issue.split(":")[0]
                problem_counts[pcode] = problem_counts.get(pcode, 0) + 1

    problems_3b = {
        "P1": "언어 전환", "P2": "내용 번역·손실",
        "P3": "JSON 생성 오류", "P4": "에이전트 오동작",
        "P5": "도구 인수 추론 실패", "P6": "컨텍스트 폭증",
    }
    print(f"\n{'문제':>4}  {'설명':<20}  {'3B':>4}  {'7B':>4}  판정")
    print("-"*60)
    for code, desc in problems_3b.items():
        count_7b = problem_counts.get(code, 0)
        verdict = "✅ 해소" if count_7b == 0 else f"❌ {count_7b}건 잔존"
        count_3b = {"P1":"빈번","P2":"전건","P3":"전건","P4":"1건","P5":"전건","P6":"1건"}[code]
        print(f"  {code}  {desc:<20}  {count_3b:>4}  {count_7b:>4}건  {verdict}")

    # 로그 저장
    log = LOG_DIR / f"7b_experiment_{timestamp}.json"
    with open(log, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_ID, "results": all_results,
                   "problem_counts": problem_counts}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  로그 저장: {log}")


if __name__ == "__main__":
    main()
