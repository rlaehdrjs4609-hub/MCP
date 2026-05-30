# MCP 시스템에서 발생하는 문제점 분석

- **분석일**: 2026-05-30
- **범위**: 한국어/영어 무관, MCP 생태계 자체에서 발생하는 문제

---

## 문제 분류

| # | 위치 | 문제 | 영향 |
|---|------|------|------|
| M1 | smolagents | null 인수 주입 | 모든 optional 파라미터 있는 도구 |
| M2 | smolagents | parse_json_blob 취약한 추출 방식 | 긴 tool call 생성 시 |
| M3 | fetch 서버 | 첫 연결 30초 타임아웃 | 첫 쿼리 항상 실패 |
| M4 | fetch 서버 | signal 파라미터 타입 불일치 | fetch 도구 호출 불가 |
| M5 | github 서버 | 공식 지원 종료 (deprecated) | 장기 사용 불가 |
| M6 | github 서버 | API 응답 과도한 메타데이터 | 컨텍스트 폭증 |
| M7 | filesystem 서버 | 스키마 ↔ 런타임 불일치 | read_text_file 호출 실패 |

---

## M1 — smolagents null 인수 주입

**위치**: `smolagents/mcp_client.py`

**현상**:
```
모델이 생성한 인수:  {path: "README.md", head: 100}
서버에 실제 전달:    {path: "README.md", head: 100, tail: null}  ← null 자동 추가
서버 오류:          "Argument tail has type 'null' but should be 'number'"
```

**실험으로 확인한 동작**:
```
직접 호출 {path: "README.md", head: 5}           → ✅ 성공
null 포함 {path: "README.md", head: 5, tail: null} → ❌ 타입 오류
```

**원인**: smolagents MCPClient가 도구 스키마의 모든 프로퍼티를 인수 딕셔너리에 포함시키면서
명시되지 않은 선택적 파라미터에 null을 채워 넣는다.

**영향**: `read_text_file`, `read_file` 등 optional number 파라미터가 있는 도구 전체.
한국어/영어 무관하게 동일하게 발생.

**수정 위치**: smolagents 소스의 tool call 변환 부분
```python
# 수정: null 값 필터링
arguments = {k: v for k, v in arguments.items() if v is not None}
```

---

## M2 — parse_json_blob 취약한 추출 방식

**위치**: `smolagents/utils.py → parse_json_blob()`

**현재 구현**:
```python
first_accolade_index = json_blob.find("{")
last_accolade_index = [a.start() for a in re.finditer("}", json_blob)][-1]
json_str = json_blob[first_accolade_index : last_accolade_index + 1]
```

**문제 1 — 여러 JSON 블록이 있을 때 잘못된 추출**:
```
모델 출력:  {"name": "tool_a"...}  설명 텍스트  {"name": "tool_b"...}
추출 결과:  {"name": "tool_a"...}  설명 텍스트  {"name": "tool_b"...}
           (첫 { 부터 마지막 } 까지 전체를 하나의 JSON으로 파싱 시도 → 실패)
```

**문제 2 — 긴 JSON에서 closing } 누락 (3B 모델 특이)**:
```
모델이 생성:  {..."observations": ["..."]}    ← outer } 누락
parse_json_blob 추출:  동일하게 }가 없는 문자열
json.loads:  실패
```

**실험 로그**:
```
Error: Expecting ',' delimiter: line 1 column 261 (char 260)
JSON blob was: {"name": "create_entities", "arguments": {...}}
```

**수정 방향**:
```python
# 괄호 균형 보완 후 파싱
opens = json_str.count("{"); closes = json_str.count("}")
if opens > closes:
    json_str += "}" * (opens - closes)
json_data = json.loads(json_str, strict=False)
```

---

## M3 — fetch 서버 첫 연결 타임아웃

**서버**: `mcp-fetch` (npm)

**현상**: 첫 번째 쿼리에서 MCP 서버 프로세스 시작 후 30초 대기 후 연결 실패.
```
오류: Couldn't connect to the MCP server after 30 seconds
```
두 번째 쿼리부터는 정상 작동.

**원인**: npm 패키지 최초 실행 시 Node.js 모듈 초기화 지연.
`@modelcontextprotocol/server-*` 계열 서버들 공통 현상.

**영향**: fetch 서버를 통한 첫 번째 쿼리는 항상 실패.
에이전트가 실패를 재시도하지 않으면 작업 전체가 포기됨.

---

## M4 — fetch 서버 signal 파라미터 타입 불일치

**서버**: `mcp-fetch`

**현상**: 모델이 fetch 도구의 `signal` 파라미터에 null을 생성하면 서버가 거부.
```
Calling tool: 'fetch' with arguments: {'url': '...', 'signal': None, ...}
→ Argument signal has type 'null' but should be 'string'
```
M1(null 주입)과 결합되면 더 심각: 모델이 명시하지 않아도 null이 주입됨.

**동일 패턴 4회 반복**: 모델이 오류를 받아도 동일한 인수로 재호출.

---

## M5 — github 서버 공식 지원 종료

**서버**: `@modelcontextprotocol/server-github@2025.4.8`

**npm 경고**:
```
npm warn deprecated @modelcontextprotocol/server-github@2025.4.8:
Package no longer supported. Contact Support at https://www.npmjs.com/support
```

**영향**:
- 보안 패치 미제공
- 신규 GitHub API 기능 미지원
- `list_pulls` 등 일부 도구의 필수 인수 오류 발생

**대안**: `github/github-mcp-server` (GitHub 공식 서버)로 교체 필요.

---

## M6 — github 서버 API 응답 과도한 메타데이터

**현상**: GitHub API가 SHA, URL, node_id 등 불필요한 필드를 대량 포함한 JSON 반환.

```
1회 API 응답 예시 (커밋 1개):
  sha, node_id, commit.author, commit.committer, commit.tree.sha, url,
  html_url, comments_url, author.login, author.id, author.avatar_url,
  author.followers_url, author.following_url, author.gists_url, ...
```

**토큰 폭증 측정**:
```
Step 1: 9,454 tokens
Step 2: 20,799 tokens  (+11,345)
Step 3: 33,905 tokens  (+13,106)
Step 4: 47,145 tokens  (+13,240)
```
커밋 1건 조회에 ~13,000 토큰 추가. 3건이면 ~40,000 토큰.

**원인**: MCP 서버가 GitHub REST API 응답 전체를 필터링 없이 전달.
필요한 정보(message, date, author.name)만 추출해서 전달하지 않음.

**영향**: 컨텍스트 한계 초과, 응답 지연, 비용 증가.

---

## M7 — filesystem 서버 스키마 ↔ 런타임 불일치

**서버**: `@modelcontextprotocol/server-filesystem`

**JSON 스키마** (`required: ["path"]`, head/tail은 선택):
```json
{
  "properties": {
    "path": {"type": "string"},
    "head": {"description": "first N lines", "type": "number"},
    "tail": {"description": "last N lines", "type": "number"}
  },
  "required": ["path"]
}
```

**실제 런타임 동작**:
```
{path: "README.md"}              → ✅ (직접 호출 시)
{path: "README.md", head: 5}     → ✅ (직접 호출 시)
{path: "README.md", head: null}  → ❌ "Invalid argument"  (M1과 결합)
```

**스키마 문제**: `head`와 `tail`이 선택 사항이라고 명시했지만,
M1(null 주입)과 결합되면 실질적으로 동작하지 않는다.
또한 head/tail을 동시에 사용하면 오류가 나지만 스키마에 이런 제약이 없다.

---

## 문제 영향 매트릭스

| 문제 | filesystem | fetch | github | memory |
|------|-----------|-------|--------|--------|
| M1 null 주입 | 🔴 주요 실패 원인 | 🔴 동일 | 🟡 일부 도구 | ✅ 영향 없음 |
| M2 JSON 파싱 | 🟡 3B에서 발생 | 🟡 | 🟡 | 🔴 주요 실패 원인 |
| M3 연결 타임아웃 | ✅ | 🔴 첫 쿼리 실패 | ✅ | ✅ |
| M4 signal 오류 | ✅ | 🔴 반복 실패 | ✅ | ✅ |
| M5 deprecated | ✅ | ✅ | 🟠 장기적 | ✅ |
| M6 응답 비대 | ✅ | ✅ | 🔴 컨텍스트 폭증 | ✅ |
| M7 스키마 불일치 | 🔴 read_text_file | ✅ | ✅ | ✅ |

---

## 요약

MCP 생태계 전반에 걸쳐 **7가지 시스템 레벨 문제**가 확인됐다.

- **smolagents 레벨** (M1, M2): 라이브러리 패치로 해결 가능
- **MCP 서버 레벨** (M3, M4, M5, M6, M7): 서버 교체 또는 업스트림 기여 필요
- 모든 문제가 **한국어/영어 무관**하게 발생

한국어 쿼리는 이 MCP 문제들을 **더 자주 노출**시키는 역할을 했지만,
근본 원인은 MCP 시스템 자체에 있다.
