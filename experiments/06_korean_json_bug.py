"""
한국어 JSON 버그 집중 분석
가설: 모델이 Korean 텍스트 포함 JSON 생성 시 raw output에 literal newline을 삽입해 파싱 실패
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# memory 서버의 create_entities 도구 스키마 (실제 실험과 동일)
SYSTEM_PROMPT = """You are a helpful assistant with access to the following tool:

create_entities(entities: list) - Create multiple entities in the knowledge graph.
  entities: list of {name: str, entityType: str, observations: list[str]}

To call a tool, output:
<tool_call>
{"name": "create_entities", "arguments": {"entities": [...]}}
</tool_call>
"""

TEST_CASES = [
    {
        "id": "en_short",
        "label": "영어 - 짧은 텍스트",
        "query": 'Store: MCP is a protocol. Qwen is an LLM by Alibaba.',
    },
    {
        "id": "en_long",
        "label": "영어 - 긴 텍스트",
        "query": 'Store: MCP is a Model Context Protocol that connects AI language models to external tools and data sources. Qwen is a multilingual Large Language Model developed by Alibaba Cloud.',
    },
    {
        "id": "ko_short",
        "label": "한국어 - 짧은 텍스트",
        "query": '저장해줘: MCP는 프로토콜이야. Qwen은 LLM이야.',
    },
    {
        "id": "ko_medium",
        "label": "한국어 - 실험과 동일한 텍스트",
        "query": '다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야.',
    },
    {
        "id": "ko_long",
        "label": "한국어 - 긴 텍스트",
        "query": '저장해줘: MCP(모델 컨텍스트 프로토콜)는 대형 언어 모델이 외부 도구 및 데이터 소스에 접근할 수 있게 해주는 표준 프로토콜입니다. Qwen은 Alibaba Cloud가 개발한 다국어 대형 언어 모델로 한국어, 중국어, 영어 등을 지원합니다.',
    },
    {
        "id": "mixed",
        "label": "혼합 - 한국어+영어 JSON",
        "query": 'Store in memory: MCP는 프로토콜. Qwen is an LLM.',
    },
]


def analyze_raw_output(raw: str, label: str):
    """raw 모델 출력에서 JSON blob을 추출해 문제점 분석"""
    result = {
        "label": label,
        "raw_length": len(raw),
        "has_tool_call": "<tool_call>" in raw,
        "literal_newline_in_json": False,
        "json_valid": False,
        "json_error": None,
        "json_blob": None,
        "char_at_error": None,
    }

    # <tool_call> 추출
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
    if not m:
        # tool_call 없으면 JSON blob 직접 찾기
        m = re.search(r'(\{\"name\".*\})', raw, re.DOTALL)
        if not m:
            result["json_error"] = "tool_call 태그 없음"
            return result

    blob = m.group(1).strip()
    result["json_blob"] = blob

    # literal newline 포함 여부 확인
    result["literal_newline_in_json"] = "\n" in blob

    # 줄별 길이 분석
    lines = blob.split("\n")
    result["line_count"] = len(lines)
    result["line_lengths"] = [len(l) for l in lines]

    # JSON 파싱 시도
    try:
        json.loads(blob)
        result["json_valid"] = True
    except json.JSONDecodeError as e:
        result["json_error"] = str(e)
        # 오류 위치 문자 확인
        pos = e.pos
        result["char_at_error"] = repr(blob[max(0,pos-10):pos+10])

    return result


def run():
    print(f"모델 로딩: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    model.eval()

    all_results = []

    for tc in TEST_CASES:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": tc["query"]},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

        raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        res = analyze_raw_output(raw, tc["label"])
        res["id"] = tc["id"]
        all_results.append(res)

        print(f"\n{'='*60}")
        print(f"[{tc['label']}]")
        print(f"  tool_call 있음: {res['has_tool_call']}")
        print(f"  JSON 내 literal newline: {res['literal_newline_in_json']}")
        print(f"  줄 수 / 각 줄 길이: {res.get('line_count')} / {res.get('line_lengths')}")
        print(f"  JSON 유효: {res['json_valid']}")
        if res["json_error"]:
            print(f"  오류: {res['json_error']}")
            print(f"  오류 위치 문자: {res['char_at_error']}")
        if res["json_blob"]:
            print(f"  JSON blob (첫 200자):")
            print(f"    {repr(res['json_blob'][:200])}")

    # 요약
    print(f"\n\n{'#'*60}")
    print("  분석 요약")
    print(f"{'#'*60}")
    print(f"{'ID':<12} {'JSON유효':>8} {'Newline':>8} {'오류위치':>30}")
    print("-"*60)
    for r in all_results:
        print(f"{r['id']:<12} {str(r['json_valid']):>8} {str(r['literal_newline_in_json']):>8} {str(r.get('char_at_error',''))[:30]:>30}")

    # JSON blob 비교 (유효한 것 vs 실패한 것)
    print("\n[핵심 비교] 영어 vs 한국어 raw JSON blob:")
    for r in all_results:
        if r["json_blob"]:
            kr_ratio = sum(1 for c in r["json_blob"] if "가"<=c<="힣") / len(r["json_blob"])
            print(f"\n  [{r['label']}]")
            print(f"  한국어 비율: {kr_ratio:.1%}")
            print(f"  repr: {repr(r['json_blob'][:300])}")

    # 저장
    import json as _json
    from pathlib import Path
    log = Path(__file__).parent.parent / "logs" / "korean_json_bug_analysis.json"
    with open(log, "w", encoding="utf-8") as f:
        _json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n로그 저장: {log}")


if __name__ == "__main__":
    run()
