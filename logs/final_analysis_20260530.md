# 한국어 JSON 버그 최종 종합 분석

- **분석일**: 2026-05-30
- **실험 범위**: 버그 재현 → 토큰 레벨 분석 → 수정 검증 → 3B vs 7B 비교

---

## 전체 실험 흐름

```
1. 대조 실험 (EN vs KO)  →  성공률 동일, 하지만 memory 서버만 KO 특이 실패
2. JSON 파싱 소스 분석   →  parse_json_blob: first{ ~ last} 추출
3. 토큰 레벨 분석        →  마지막 토큰 패턴이 모델 크기별로 다름
4. 임계값 검증           →  JSON ~260자 이상에서 잘못된 토큰 선택
5. 수정 효과 검증        →  structured_output=False 우회 확인
6. 3B vs 7B 비교         →  7B는 올바른 토큰 패턴 사용
```

---

## 근본 원인 (확정)

### 버그: Qwen2.5-3B의 긴 tool call JSON에서 outer `}` 누락

모델이 `<tool_call>` 텍스트 포맷으로 JSON을 생성할 때, JSON 길이가 **~260자 이상**이 되면 잘못된 마지막 토큰을 선택한다.

```
올바른 패턴 (짧은 JSON):   ... ']}'  →  '}\n'   →  </tool_call>
잘못된 패턴 (긴 JSON):     ... ']}'  →  ']}\n'  →  </tool_call>
```

- `'}\n'` : arguments 닫기 + outer wrapper 닫기 + 개행 → **4개 `{` = 4개 `}`** ✅
- `']}\n'` : entities array + arguments 닫기 + 개행, **outer wrapper 누락** → **4개 `{` ≠ 3개 `}`** ❌

### 한국어와의 연결 고리

한국어 입력(짧은 설명) → 모델이 영어로 번역·확장하여 긴 관찰값 생성 → JSON 260자 초과 → 잘못된 토큰 패턴 트리거

---

## 3B vs 7B 토큰 패턴 비교

동일 쿼리: `"다음 정보를 메모리에 저장해줘: MCP는... Qwen은..."`

| 항목 | Qwen2.5-3B | Qwen2.5-7B |
|------|-----------|-----------|
| JSON 유효 | ❌ | ✅ |
| JSON 길이 | 306자 (outer `}` 포함 시 307) | 274자 |
| 마지막 토큰 -2 | `']}\n'` ← 버그 | `'}}\n'` ← 올바름 |
| 생성 토큰 수 | 82 | 74 |
| 관찰값 (MCP) | "MCP is a protocol that connects AI models and external tools." (60자) | "AI model and external tools connecting protocol" (47자) |
| 관찰값 (Qwen) | "Qwen is an AI language model developed by Alibaba." (50자) | "Alibaba-developed multilingual language model" (45자) |

7B 모델은:
1. 관찰값을 더 **간결하게** 생성 → JSON이 짧아짐
2. `'}}\n'` 토큰 패턴 학습 → outer `}` 포함한 올바른 JSON 생성

---

## 수정 방법 효과 요약

| 방법 | memory 버그 해결 | 한국어 관찰값 유지 | 부작용 |
|------|----------------|-----------------|--------|
| `MCPClient(so=False)` | ✅ | ❌ (영어로 변환) | github 에이전트 오동작 위험 |
| `MCPClient(so=True)` + 짧은 입력 | ✅ | ✅ | 긴 입력 시 버그 재발 |
| 7B 모델 교체 | ✅ | ✅ | VRAM 14GB 필요 |
| parse_json_blob 후처리 (`}` 보완) | ✅ | ✅ | smolagents 내부 수정 필요 |
| 시스템 프롬프트 간결성 지시 | ⚠️ 불안정 | ⚠️ | PromptTemplates API 호환성 문제 |

---

## 최종 권고사항

### 단기 (즉시 적용)
```python
# 방법 1: so=False (native function calling)
with MCPClient(server_params, structured_output=False) as tools:
    agent = ToolCallingAgent(tools=tools, model=model, max_steps=3)
```

### 중기 (권장)
```python
# 방법 2: 7B 모델로 교체
model = TransformersModel(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,   # ~14GB VRAM
)
```

### 근본 해결 (라이브러리 기여 가능)
smolagents의 `parse_json_blob` 함수에 괄호 균형 보완 로직 추가:
```python
# 수정 위치: smolagents/utils.py parse_json_blob()
opens = json_str.count("{"); closes = json_str.count("}")
if opens > closes:
    json_str += "}" * (opens - closes)
json_data = json.loads(json_str, strict=False)
```

---

## 실험 데이터 파일 목록

| 파일 | 내용 |
|------|------|
| `logs/analysis_20260530.md` | MCP 서버별 최초 실험 결과 |
| `logs/comparison_analysis_20260530.md` | EN vs KO 대조 실험 분석 |
| `logs/korean_json_bug_rootcause.md` | 버그 근본 원인 상세 분석 |
| `logs/korean_json_bug_analysis.json` | 06번 실험 raw 데이터 |
| `logs/json_truncation_test.json` | 07번 절단 지점 검증 데이터 |
| `logs/fix_verification.json` | 08번 수정 효과 검증 데이터 |
| `logs/final_analysis_20260530.md` | 본 문서 (종합 결론) |
