# 해소되지 않은 문제 심층 분석: P1 & P5

- **분석일**: 2026-05-30
- **모델**: Qwen/Qwen2.5-7B-Instruct
- **실험**: filesystem 서버 (P1: 언어 전환, P5: 도구 인수 추론 실패)

---

## P1 — 언어 전환 (한국어 쿼리 → 중국어/영어 응답)

### 발생 패턴 실험 결과

| 쿼리 유형 | 실제 응답 언어 | 응답 앞부분 |
|-----------|--------------|------------|
| `"현재 디렉토리에 있는 파일 목록을 알려줘."` | 🔴 **중국어** | 当前目录下的文件列表如下 |
| `"파일 목록?"` (초단문) | 🔴 **중국어** | 以下是当前目录中的文件列表 |
| `"README.md 파일 내용을 요약해줘."` | ✅ 한국어 | 죄송합니다, 현재 README.md ... |
| `"한국어로만 대답해줘. 파일 목록을 알려줘."` | ✅ 한국어 | 현재 디렉토리의 파일 목록 |
| `"Please list files. 한국어로 답해줘."` | ✅ 한국어 | 현재 디렉토리 내의 파일 목록 |
| `"...파일과 폴더 목록을 한국어로 상세히 알려주시겠어요?"` | ✅ 한국어 | 현재 디렉토리에는 다음과 같은 |

### 트리거 조건 분석

```
중국어 전환 발생:
  - 쿼리에 "한국어로" 명시 없음
  - 쿼리가 짧음 (3~10자)

한국어 유지:
  - 쿼리에 "한국어로" 또는 "한국어로만" 포함
  - 영어+한국어 혼합이라도 "한국어로" 명시하면 유지
  - 쿼리가 길고 정중한 문체 (모델이 더 신중하게 처리)
```

### 왜 중국어인가?

Qwen 모델은 Alibaba가 개발한 중국어 중심 모델이다.
한국어가 입력되어도 모델 내부에서는 한국어/중국어 구분이 불명확할 수 있다.
특히 **짧은 한국어 쿼리**는 중국어 컨텍스트로 처리될 가능성이 높다.

```
"파일 목록?"  →  모델이 이를 중국어 쿼리로 인식
→ 以下是当前目录中的文件列表 (중국어 응답)
```

### 시스템 프롬프트 효과

| 방식 | 실제 응답 | 효과 |
|------|----------|------|
| 지시 없음 | 중국어 | ❌ |
| 쿼리 앞에 "한국어 전용 어시스턴트" 지시 삽입 | 한국어 포함* | ⚠️ 부분 |
| 쿼리 앞에 "한국어 AI 연구자" 역할 삽입 | 한국어 포함* | ⚠️ 부분 |

*응답에 파일명(ASCII) 포함으로 언어 감지 오류 발생. 실제로는 한국어 문장이 포함됨.

### 근본 원인

**Qwen 모델의 중국어 편향**: Qwen2.5-7B는 중국어 학습 데이터가 압도적으로 많아,
짧거나 언어 지시가 없는 한국어 쿼리를 중국어 컨텍스트로 처리함.

**해결책**:
1. **즉시 적용**: 모든 쿼리에 `"반드시 한국어로 답해줘."` 접미어 추가
2. **근본 해결**: 한국어 특화 모델 사용 (EXAONE 3.0, HyperCLOVA X, Llama-3-Korean)

---

## P5 — 도구 인수 추론 실패 (filesystem read_text_file)

### 실제 스키마 vs 실제 동작

**JSON 스키마** (`required: ["path"]`):
```json
{
  "properties": {
    "path": {"type": "string"},
    "head": {"description": "first N lines", "type": "number"},
    "tail": {"description": "last N lines", "type": "number"}
  },
  "required": ["path"]   ← head/tail은 선택 사항!
}
```

**직접 MCP 호출 결과**:
| 인수 조합 | 직접 호출 | smolagents 에이전트 |
|-----------|-----------|-------------------|
| `{path}` only | ✅ 파일 전체 내용 반환 | ❌ "tail is required" |
| `{path, tail: 5}` | ✅ 마지막 5줄 | ❌ "head is required" |
| `{path, head: 5}` | ✅ 처음 5줄 | ❌ "tail is required" |
| `{path, head: 5, tail: 5}` | ✅ (오류 메시지 반환) | ❌ "Cannot specify both" |

### 핵심 발견: smolagents가 null 인수를 주입한다

```
모델이 생성하는 tool call:
  read_text_file(path='README.md', head=100)
              ↓ smolagents MCPClient 처리
실제 서버에 전달되는 인수:
  {path: 'README.md', head: 100, tail: null}  ← null 자동 주입!
              ↓ MCP 서버 타입 검증
오류:
  "Argument tail has type 'null' but should be 'number'"
```

smolagents의 `MCPClient`가 **도구 스키마의 모든 선택적 프로퍼티를 null로 채워** 전달한다.
MCP 서버는 null을 number 타입으로 허용하지 않아 검증 실패.

### 직접 호출이 성공하는 이유

직접 `session.call_tool("read_text_file", {"path": "README.md"})` 호출 시
→ `tail` 키 자체가 없음 → 서버가 선택 사항으로 처리 → ✅ 성공

smolagents를 통한 호출 시
→ `{path: ..., head: ..., tail: null}` → 서버가 null 타입 거부 → ❌ 실패

### 힌트 제공 효과

어떤 힌트를 줘도 모두 실패:
- `"tail에 숫자를 넣어야 해"` → 모델이 tail을 넣으면 head=null 문제로 동일 오류
- `"list_directory로 대체"` → 파일 내용 읽기 불가능
- `"read_file 사용해봐"` → read_file도 동일한 null 주입 문제

### 우회 방법

**방법 1**: `read_multiple_files` 사용 (head/tail 인수 없음)
```
read_multiple_files(paths=['README.md'])  → ✅ 작동
```

**방법 2**: smolagents MCPClient null 주입 비활성화
smolagents 소스코드(`mcp_client.py`)에서 null 값 필터링:
```python
# 수정 필요 위치: MCPClient의 tool call 변환 부분
arguments = {k: v for k, v in arguments.items() if v is not None}
```

**방법 3**: MCP 서버 업데이트 대기
`@modelcontextprotocol/server-filesystem`이 null을 '미지정'으로 처리하도록 수정

---

## 두 문제의 공통 특성

| | P1 (언어 전환) | P5 (도구 인수 실패) |
|--|----------------|-------------------|
| 한국어 특이 문제? | ✅ 예 | ❌ 아님 (EN도 동일) |
| 모델 크기 해결? | ❌ 7B도 동일 | ❌ 7B도 동일 |
| 근본 원인 | Qwen 중국어 편향 | smolagents null 주입 버그 |
| 즉시 해결책 | 쿼리에 "한국어로" 명시 | `read_multiple_files` 사용 |
| 근본 해결책 | 한국어 특화 모델 | smolagents 소스 패치 |
