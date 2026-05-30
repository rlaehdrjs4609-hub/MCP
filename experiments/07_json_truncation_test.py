"""
JSON 절단(truncation) 가설 검증
가설: 모델이 tool call JSON을 ~260자에서 잘라냄
     → 한국어가 영어보다 chars/token 비율이 높아 한국어 JSON이 먼저 한계에 도달

검증 방법:
1. 관찰값 텍스트 길이를 점진적으로 늘리면서 생성된 JSON 길이 측정
2. 영어 vs 한국어에서 절단 지점이 동일한지 확인
3. smolagents parse_json_blob 로직 직접 시뮬레이션
"""

import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# smolagents의 실제 파싱 로직 재현
def parse_json_blob(json_blob: str):
    try:
        first = json_blob.find("{")
        last  = [m.start() for m in re.finditer("}", json_blob)][-1]
        json_str = json_blob[first:last+1]
        return json.loads(json_str, strict=False), json_str
    except IndexError:
        return None, "NO_BRACE"
    except json.JSONDecodeError as e:
        return None, f"PARSE_ERROR at char {e.pos}: {json_blob[max(0,e.pos-5):e.pos+5]!r}"


# smolagents 실제 tool call 프롬프트 포맷 재현
TOOL_DESC = {
    "type": "function",
    "function": {
        "name": "create_entities",
        "description": "Create entities in knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entityType": {"type": "string"},
                            "observations": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
    }
}


def make_query_and_expected_json(lang, obs_len):
    """관찰값 텍스트 길이를 조절한 테스트 케이스 생성"""
    if lang == "en":
        obs = "A" * obs_len  # 영어 ASCII 패딩
        query = f"Store in memory: entity1 obs={obs}"
    else:
        obs = "가" * obs_len  # 한국어 글자 패딩 (3 bytes each)
        query = f"메모리에 저장: entity1 관찰={obs}"

    expected_len = len(
        f'{{"name": "create_entities", "arguments": {{"entities": [{{"name": "entity1", "entityType": "test", "observations": ["{obs}"]}}]}}}}'
    )
    return query, obs, expected_len


def run():
    print(f"모델 로딩: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16
    )
    model.eval()

    results = []

    # 관찰값 길이를 10~80자씩 테스트 (한국어는 bytes로 3배)
    for lang in ["en", "ko"]:
        for obs_len in [10, 20, 30, 40, 50, 60, 70, 80]:
            query, obs, expected_json_len = make_query_and_expected_json(lang, obs_len)

            msgs = [{"role": "user", "content": query}]
            # smolagents 실제 chat template (tools 파라미터 포함)
            text = tok.apply_chat_template(
                msgs,
                tools=[TOOL_DESC],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tok([text], return_tensors="pt").to("cuda:0")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=400, do_sample=False)

            raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # smolagents parse_json_blob 시뮬레이션
            parsed, json_str = parse_json_blob(raw)

            actual_json_len = len(json_str) if isinstance(json_str, str) and json_str not in ("NO_BRACE",) and not json_str.startswith("PARSE_ERROR") else 0
            success = parsed is not None

            result = {
                "lang": lang,
                "obs_len": obs_len,
                "obs_bytes": len(obs.encode("utf-8")),
                "expected_json_chars": expected_json_len,
                "actual_json_chars": actual_json_len,
                "success": success,
                "error": json_str if not success else None,
                "literal_newline": "\n" in raw,
                "has_tool_call_tag": "<tool_call>" in raw,
            }
            results.append(result)

            status = "✅" if success else "❌"
            print(f"[{lang}] obs_len={obs_len:3d} bytes={result['obs_bytes']:4d} | "
                  f"expected_json={expected_json_len:4d} actual_json={actual_json_len:4d} | "
                  f"{status} | newline={result['literal_newline']}")

    # 분석 출력
    print(f"\n\n{'='*70}")
    print("  절단 지점 분석: 영어 vs 한국어")
    print(f"{'='*70}")
    print(f"{'lang':>4} {'obs_len':>8} {'obs_bytes':>10} {'exp_json':>9} {'act_json':>9} {'ok':>4} {'newline':>8}")
    print("-"*70)
    for r in results:
        s = "✅" if r["success"] else "❌"
        print(f"{r['lang']:>4} {r['obs_len']:>8} {r['obs_bytes']:>10} "
              f"{r['expected_json_chars']:>9} {r['actual_json_chars']:>9} {s:>4} {str(r['literal_newline']):>8}")

    # 절단 지점 찾기
    print("\n[결론]")
    en_fail = [r for r in results if r["lang"]=="en" and not r["success"]]
    ko_fail = [r for r in results if r["lang"]=="ko" and not r["success"]]
    en_ok   = [r for r in results if r["lang"]=="en" and r["success"]]
    ko_ok   = [r for r in results if r["lang"]=="ko" and r["success"]]

    if en_fail:
        print(f"  영어 첫 실패: obs_len={en_fail[0]['obs_len']}, exp_json={en_fail[0]['expected_json_chars']}")
    else:
        print(f"  영어: 모든 케이스 성공")
    if ko_fail:
        print(f"  한국어 첫 실패: obs_len={ko_fail[0]['obs_len']}, exp_json={ko_fail[0]['expected_json_chars']}")
    else:
        print(f"  한국어: 모든 케이스 성공")

    if en_ok:
        max_en = max(r["expected_json_chars"] for r in en_ok)
        print(f"  영어 최대 성공 JSON 길이: {max_en} chars")
    if ko_ok:
        max_ko = max(r["expected_json_chars"] for r in ko_ok)
        print(f"  한국어 최대 성공 JSON 길이: {max_ko} chars")

    # 저장
    from pathlib import Path
    log = Path(__file__).parent.parent / "logs" / "json_truncation_test.json"
    with open(log, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n로그 저장: {log}")


if __name__ == "__main__":
    run()
