"""
Smoke test for the sys.stdout/stderr UTF-8 reconfigure in agent.py.

Reproduces the crash hit earlier: Rich's legacy Windows console path
encodes through sys.stdout, and without an explicit UTF-8 reconfigure,
cp1252 can't represent characters like the arrow / em-dash / section-sign
glyphs this project's console output uses, raising UnicodeEncodeError.

Importing agent.py applies the reconfigure at module load (the same code
path a real run uses) — this test then prints the exact character classes
that previously crashed and confirms it doesn't raise. No SDK/API calls
involved, no network access, no subscription usage — pure console
rendering check.

Run with:
    python tests/test_console_encoding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402  (import after sys.path tweak; applies the stdout/stderr reconfigure)


def main() -> None:
    try:
        agent.console.print("→ § — test")  # → § — test
    except UnicodeEncodeError as exc:
        print(f"FAIL: UnicodeEncodeError still raised: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"FAIL: unexpected exception: {exc!r}")
        sys.exit(1)
    else:
        print("PASS: console.print handled arrow/section-sign/em-dash without raising")


if __name__ == "__main__":
    main()
