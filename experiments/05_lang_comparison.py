"""
실험 #5 — 영어 vs 한국어 쿼리 대조 실험
목적: 문제가 '소형 모델 한계'인지 '한국어 처리 문제'인지 분리
모델: Qwen/Qwen2.5-3B-Instruct (GPU)
MCP 서버: filesystem, github, memory (fetch는 연결 불안정으로 제외)
"""

import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import torch
from mcp import StdioServerParameters
from smolagents import ToolCallingAgent, TransformersModel
from smolagents.mcp_client import MCPClient

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# 같은 의미의 쿼리를 영어/한국어 쌍으로 정의
QUERY_PAIRS = {
    "filesystem": [
        {
            "en": "List all files and folders in the current directory.",
            "ko": "현재 디렉토리에 있는 모든 파일과 폴더 목록을 알려줘.",
        },
        {
            "en": "Read the README.md file and summarize its contents.",
            "ko": "README.md 파일을 읽고 내용을 요약해줘.",
        },
    ],
    "github": [
        {
            "en": "List the files in the rlaehdrjs4609-hub/MCP repository.",
            "ko": "rlaehdrjs4609-hub/MCP 저장소의 파일 목록을 알려줘.",
        },
        {
            "en": "Summarize the 3 most recent commits in the rlaehdrjs4609-hub/MCP repository.",
            "ko": "rlaehdrjs4609-hub/MCP 저장소의 최근 커밋 3개를 요약해줘.",
        },
    ],
    "memory": [
        {
            "en": "Store the following in memory: MCP is a protocol connecting AI models to external tools. Qwen is a multilingual LLM made by Alibaba.",
            "ko": "다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야.",
        },
        {
            "en": "Retrieve the information about MCP and Qwen from memory and explain their relationship.",
            "ko": "메모리에서 MCP와 Qwen 정보를 불러와서 두 개의 관계를 설명해줘.",
        },
    ],
}

SERVER_CONFIGS = {
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

def run_query(server_name, lang, query, model):
    entry = {
        "lang": lang,
        "query": query,
        "response": None,
        "error": None,
        "elapsed_sec": None,
        "steps": 0,
        "tool_call_errors": 0,
        "final_lang": None,
    }
    t0 = time.time()
    try:
        with MCPClient(SERVER_CONFIGS[server_name]) as tools:
            agent = ToolCallingAgent(
                tools=tools, model=model, max_steps=5,
            )
            response = agent.run(query)
            entry["response"] = str(response)[:500]

            # 응답 언어 추정
            korean_ratio = sum(1 for c in str(response) if "가" <= c <= "힣") / max(len(str(response)), 1)
            entry["final_lang"] = "ko" if korean_ratio > 0.05 else "en"
    except Exception as e:
        entry["error"] = str(e)
    entry["elapsed_sec"] = round(time.time() - t0, 1)
    return entry


def main():
    print(f"모델 로딩 중: {MODEL_ID}")
    model = TransformersModel(
        model_id=MODEL_ID,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for server_name, pairs in QUERY_PAIRS.items():
        if server_name == "github" and not GITHUB_TOKEN:
            print(f"\n[SKIP] github: GITHUB_TOKEN 미설정")
            continue

        print(f"\n{'='*60}\n  [{server_name.upper()}] 대조 실험\n{'='*60}")
        server_results = []

        for pair in pairs:
            for lang in ["en", "ko"]:
                query = pair[lang]
                print(f"\n  [{lang.upper()}] {query[:60]}")
                result = run_query(server_name, lang, query, model)
                result["query_full"] = query
                server_results.append(result)

                status = "✅" if not result["error"] else "❌"
                lang_match = ""
                if result["final_lang"] and lang == "ko":
                    lang_match = "한국어 유지 ✅" if result["final_lang"] == "ko" else "⚠️ 영어로 전환"
                print(f"  {status} {result['elapsed_sec']}초 | {lang_match}")
                if result["response"]:
                    print(f"  응답: {result['response'][:150]}")
                if result["error"]:
                    print(f"  오류: {result['error'][:150]}")

        all_results[server_name] = server_results

    # 비교 분석
    print(f"\n\n{'#'*60}")
    print("  영어 vs 한국어 대조 분석")
    print(f"{'#'*60}")

    comparison = {}
    for server, entries in all_results.items():
        en_entries = [e for e in entries if e["lang"] == "en"]
        ko_entries = [e for e in entries if e["lang"] == "ko"]

        en_success = sum(1 for e in en_entries if not e["error"])
        ko_success = sum(1 for e in ko_entries if not e["error"])
        en_time = sum(e["elapsed_sec"] for e in en_entries) / max(len(en_entries), 1)
        ko_time = sum(e["elapsed_sec"] for e in ko_entries) / max(len(ko_entries), 1)
        ko_lang_keep = sum(1 for e in ko_entries if e["final_lang"] == "ko")

        comparison[server] = {
            "en_success": f"{en_success}/{len(en_entries)}",
            "ko_success": f"{ko_success}/{len(ko_entries)}",
            "en_avg_sec": round(en_time, 1),
            "ko_avg_sec": round(ko_time, 1),
            "ko_lang_maintained": f"{ko_lang_keep}/{len(ko_entries)}",
        }

        print(f"\n  [{server}]")
        print(f"    성공률   — 영어: {en_success}/{len(en_entries)}  한국어: {ko_success}/{len(ko_entries)}")
        print(f"    평균속도  — 영어: {en_time:.1f}초  한국어: {ko_time:.1f}초")
        print(f"    언어유지  — 한국어 응답 유지: {ko_lang_keep}/{len(ko_entries)}")

    # 결론 도출
    print(f"\n  [결론]")
    for server, c in comparison.items():
        en_s = int(c["en_success"].split("/")[0])
        ko_s = int(c["ko_success"].split("/")[0])
        total = int(c["en_success"].split("/")[1])
        if en_s == ko_s == total:
            print(f"  - {server}: 영어/한국어 모두 성공 → 모델 능력 충분, 한국어 문제 없음")
        elif en_s > ko_s:
            print(f"  - {server}: 영어 성공({en_s}) > 한국어 성공({ko_s}) → 한국어 처리 문제 존재")
        elif en_s == ko_s and en_s < total:
            print(f"  - {server}: 영어/한국어 동일 실패 → 모델 능력 한계 (언어 무관)")
        else:
            print(f"  - {server}: 혼합 결과 → 추가 분석 필요")

    # 로그 저장
    log_path = LOG_DIR / f"comparison_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model": MODEL_ID,
            "device": "cuda:0",
            "results": all_results,
            "comparison": comparison,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  로그 저장: {log_path}")


if __name__ == "__main__":
    main()
