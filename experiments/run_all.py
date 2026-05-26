"""
전체 실험 순서대로 실행
"""

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    ("01_filesystem.py", "filesystem - 로컬 파일 읽기/쓰기"),
    ("02_fetch.py",      "fetch     - 웹페이지 가져오기"),
    ("03_github.py",     "github    - GitHub 저장소 조회"),
    ("04_memory.py",     "memory    - 지식 그래프 메모리"),
]

def main():
    base = Path(__file__).parent

    for filename, desc in EXPERIMENTS:
        print(f"\n{'#'*60}")
        print(f"  실험: {desc}")
        print(f"{'#'*60}")

        result = subprocess.run(
            [sys.executable, str(base / filename)],
            check=False,
        )

        if result.returncode != 0:
            print(f"[!] {filename} 실패 (returncode={result.returncode}), 다음 실험으로 넘어갑니다.")

if __name__ == "__main__":
    main()
