"""
MCP 서버별 한국어 쿼리 실험 + 문제점 분석
결과는 logs/ 폴더에 저장
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

EXPERIMENTS = {
    "filesystem": {
        "server": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem",
                  str(Path(__file__).parent.parent)],
        ),
        "queries": [
            "현재 디렉토리에 어떤 파일과 폴더가 있는지 한국어로 설명해줘",
            "README.md 파일을 읽고 내용을 한국어로 요약해줘",
        ],
    },
    "fetch": {
        "server": StdioServerParameters(
            command="npx",
            args=["-y", "mcp-fetch"],
        ),
        "queries": [
            "https://modelcontextprotocol.io 페이지를 가져와서 MCP가 무엇인지 한국어로 설명해줘",
            "https://github.com/modelcontextprotocol/servers 페이지를 가져와서 주요 MCP 서버 목록을 한국어로 정리해줘",
        ],
    },
    "github": {
        "server": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", "")},
        ),
        "queries": [
            "rlaehdrjs4609-hub/MCP 저장소의 파일 목록을 한국어로 알려줘",
            "rlaehdrjs4609-hub/MCP 저장소의 최근 커밋 3개를 한국어로 요약해줘",
        ],
    },
    "memory": {
        "server": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
        ),
        "queries": [
            "다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야.",
            "아까 저장한 MCP와 Qwen 정보를 불러와서 두 개의 관계를 한국어로 설명해줘",
        ],
    },
}


def run_single(name, config, model):
    results = []
    for query in config["queries"]:
        entry = {"query": query, "response": None, "error": None,
                 "elapsed_sec": None, "tool_calls": []}
        print(f"\n  쿼리: {query}")
        t0 = time.time()
        try:
            with MCPClient(config["server"]) as tools:
                agent = ToolCallingAgent(
                    tools=tools, model=model, max_steps=4,
                )
                # 도구 호출 추적을 위해 콜백 활용
                response = agent.run(query)
                entry["response"] = str(response)
                print(f"  응답: {str(response)[:300]}")
        except Exception as e:
            entry["error"] = traceback.format_exc()
            print(f"  [오류] {e}")
        entry["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(entry)
    return results


def analyze_issues(all_results):
    """실험 결과에서 한국어 쿼리 문제점을 자동 분석"""
    issues = []

    for server, entries in all_results.items():
        for e in entries:
            if e["error"]:
                issues.append({
                    "server": server,
                    "type": "tool_error",
                    "query": e["query"],
                    "detail": e["error"].splitlines()[-1],
                })
                continue

            resp = e.get("response") or ""

            # 한국어 응답 비율 체크
            korean_chars = sum(1 for c in resp if "가" <= c <= "힣")
            if len(resp) > 50 and korean_chars / max(len(resp), 1) < 0.05:
                issues.append({
                    "server": server,
                    "type": "language_switch",
                    "query": e["query"],
                    "detail": "한국어 쿼리에 영어로 응답 (한국어 비율 5% 미만)",
                })

            # 응답이 너무 짧거나 없음
            if len(resp.strip()) < 20:
                issues.append({
                    "server": server,
                    "type": "empty_response",
                    "query": e["query"],
                    "detail": f"응답이 너무 짧음 ({len(resp.strip())}자)",
                })

    return issues


def main():
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("[경고] GITHUB_TOKEN 미설정 — github 실험 건너뜀")

    model = TransformersModel(
        model_id=MODEL_ID, device_map="auto", torch_dtype="auto",
    )

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, config in EXPERIMENTS.items():
        if name == "github" and not github_token:
            all_results[name] = [{"query": q, "response": None,
                                   "error": "GITHUB_TOKEN 미설정", "elapsed_sec": 0}
                                  for q in config["queries"]]
            continue

        print(f"\n{'='*60}\n  [{name.upper()}] MCP 서버 실험\n{'='*60}")
        all_results[name] = run_single(name, config, model)

    # 문제점 분석
    issues = analyze_issues(all_results)

    # 로그 저장
    log_path = LOG_DIR / f"experiment_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "model": MODEL_ID,
                   "results": all_results, "issues": issues}, f,
                  ensure_ascii=False, indent=2)

    # 분석 요약 출력
    print(f"\n\n{'#'*60}")
    print("  한국어 쿼리 문제점 분석 결과")
    print(f"{'#'*60}")

    if not issues:
        print("  발견된 문제 없음")
    else:
        by_type = {}
        for iss in issues:
            by_type.setdefault(iss["type"], []).append(iss)

        type_labels = {
            "tool_error": "도구 호출 오류",
            "language_switch": "언어 전환 문제 (한→영)",
            "empty_response": "응답 누락/부실",
        }
        for t, items in by_type.items():
            print(f"\n  [{type_labels.get(t, t)}] {len(items)}건")
            for item in items:
                print(f"    - [{item['server']}] {item['detail']}")
                print(f"      쿼리: {item['query'][:60]}")

    print(f"\n  로그 저장: {log_path}")
    print(f"{'#'*60}")

    return log_path


if __name__ == "__main__":
    main()
