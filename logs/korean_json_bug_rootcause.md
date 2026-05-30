# 한국어 JSON 버그 근본 원인 분석

- **분석일**: 2026-05-30
- **모델**: Qwen/Qwen2.5-3B-Instruct
- **프레임워크**: smolagents 1.25.0
- **재현 서버**: memory (`@modelcontextprotocol/server-memory`)

---

## 결론 요약

> **한국어 입력이 긴 JSON을 유발하고, 3B 모델이 긴 JSON 생성 시 마지막 `}` 토큰을 누락하는 훈련 아티팩트(training artifact)가 버그의 근본 원인이다.**

---

## 1. 버그 재현

**입력**: `"다음 정보를 메모리에 저장해줘: MCP는 AI 모델과 외부 도구를 연결하는 프로토콜이야. Qwen은 Alibaba가 만든 다국어 LLM이야."`

**실제 생성된 raw output**:
```
<tool_call>
{"name": "create_entities", "arguments": {"entities": [
  {"name": "MCP", "entityType": "AI Protocol", "observations": ["MCP is a protocol..."]},
  {"name": "Qwen", "entityType": "Language Model", "observations": ["Qwen is an AI...Alibaba."]}
]}\n</tool_call>
```

**parse_json_blob 추출 결과 (306 chars)**:
```
{"name": "create_entities", "arguments": {...}}  ← { : 4개, } : 3개 → 차이 = 1
```
→ 마지막 `}` 누락 → `json.loads` 실패

---

## 2. 토큰 레벨 분석 (핵심 증거)

모델의 마지막 4개 생성 토큰을 케이스별 비교:

| 케이스 | JSON 유효 | 토큰 -3 | 토큰 -2 | 토큰 -1 |
|--------|----------|---------|---------|---------|
| EN 짧음 (192자) | ✅ | `']}'` | `'}\n'` | `</tool_call>` |
| EN 중간 (268자) | ✅ | `']}'` | `'}\n'` | `</tool_call>` |
| KO 짧음 (138자) | ✅ | `']'` | `'}}\n'` | `</tool_call>` |
| **KO 실제실패 (306자)** | **❌** | `']}'` | **`']}\n'`** | `</tool_call>` |

**성공 패턴**: 마지막에 `'}\n'` 또는 `'}}\n'` → 외부 wrapper `}` 포함 ✅  
**실패 패턴**: 마지막에 **`']}\n'` (token id=23439)** → entities 배열`]` + arguments `}` + `\n` 을 하나의 토큰으로 처리, 외부 wrapper `}` 누락 ❌

---

## 3. 버그 메커니즘

```
JSON 구조:
{                                  ← outer wrapper (create_entities)
  "name": "...",
  "arguments": {                   ← arguments
    "entities": [                  ← entities array
      {...},                       ← entity1
      {...}                        ← entity2
    ]          ← ']'
  }            ← '}' (arguments)
}              ← '}' (outer wrapper) ← 이 토큰이 누락됨
```

**성공 시**: `']}'` (entity2+array 닫기) → `'}\n'` (arguments `}` + outer `}` + newline)  
**실패 시**: `']}'` (entity2 닫기) → `']}\n'` (array+arguments `}` + newline) → outer `}` 생략

token id=23439 (`']}\n'`)이 생성되는 순간 outer wrapper가 닫히지 않는다.

---

## 4. 길이 임계값 (threshold)

| JSON 길이 | 결과 | 비고 |
|-----------|------|------|
| ~138자 (KO 짧음) | ✅ | `'}}\n'` 패턴 사용 |
| ~192자 (EN 짧음) | ✅ | `'}\n'` 패턴 사용 |
| ~268자 (EN 중간) | ✅ | `'}\n'` 패턴 사용 |
| **~260자 (실험 KO 실패)** | **❌** | `']}\n'` 패턴 |
| **~306자 (재현 KO 실패)** | **❌** | `']}\n'` 패턴 |

**임계값**: JSON 길이 약 260자 이상에서 잘못된 토큰 패턴이 나타남.

---

## 5. 한국어와의 관계

한국어가 '직접적인' 인코딩 오류를 일으키는 것이 아니다.  
한국어 입력 → 모델이 영어로 번역/확장하여 observations 생성 → JSON 길이 증가 → 임계값 초과 → 버그 발생

```
"MCP는 프로토콜이야" (8자)
    ↓ 모델이 영어로 elaboration
"MCP is a protocol that connects AI models to external tools." (60자)
```

즉 **한국어 입력 자체가 문제가 아니라, 짧은 한국어 설명을 더 긴 영어 문장으로 번역하는 모델의 행동이 JSON 길이를 임계값 이상으로 만드는 것**이다.

---

## 6. structured_output 파라미터 효과

| structured_output (MCPClient) | 결과 | 동작 방식 |
|-------------------------------|------|-----------|
| `False` (구 기본값) | ✅ **성공** | Qwen 네이티브 function calling → 짧고 간결한 arguments 생성, `<tool_call>` 텍스트 포맷 미사용 |
| `True` (신 기본값) | ❌ **실패** | `<tool_call>` 텍스트 포맷 → 동일한 missing `}` 버그 |

→ `structured_output=False`는 Qwen 모델의 native function call 형식을 활용하여 bug를 우회한다.  
단, 이 경우에도 observations를 영어로 번역하는 동작은 그대로이다.

---

## 7. 수정 방법 (우선순위 순)

### 방법 1 — MCPClient에 `structured_output=False` 명시 (즉각 적용 가능)
```python
with MCPClient(server_params, structured_output=False) as tools:
    agent = ToolCallingAgent(tools=tools, model=model, max_steps=3)
    result = agent.run(query)
```
smolagents 1.25에서 기본값이 `True`로 바뀌기 전에 명시적으로 지정.

### 방법 2 — parse_json_blob 후처리 (라이브러리 수준)
```python
# JSON 파싱 전 균형 검사 후 보완
opens = blob.count("{"); closes = blob.count("}")
if opens > closes:
    blob += "}" * (opens - closes)
json.loads(blob, strict=False)
```
보완 후 파싱 성공 검증 완료 (`}` 추가만으로 정상 파싱됨).

### 방법 3 — 시스템 프롬프트에 간결성 지시 추가
```
"observations must be concise (under 50 chars each)"
```
JSON 길이를 임계값 이하로 제어.

### 방법 4 — 모델 업그레이드 (7B 이상)
더 큰 모델은 tool call 생성 정확도가 높아 동일 버그가 발생하지 않을 가능성 높음.  
RTX A6000 49GB VRAM으로 7B bfloat16(~14GB) 충분히 수용 가능.

---

## 8. 다음 실험 제안

1. **7B 모델 동일 쿼리 테스트**: Qwen2.5-7B-Instruct로 같은 KO 쿼리 실행
2. **시스템 프롬프트 효과 측정**: "observations는 50자 이내로 간결하게" 추가 후 성공률 비교
3. **다른 MCP 서버 확장**: filesystem의 `read_text_file` `tail` 인수 오류도 동일한 structured_output=False로 개선되는지 확인
