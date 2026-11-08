"""Ask the coach chat about a player's stored sessions, in the terminal (Phase 9, DEC-043).

    uv run python tools/coach_chat.py --player me                 # interactive
    uv run python tools/coach_chat.py --player me "How was my last kick?"

Sessions come from the app's database (data/kickcoach.db: upload kicks at
http://127.0.0.1:8000/ first). Model turns are cached in data/cache/chat/. Without a
GEMINI_API_KEY every answer is a template listing the latest session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kickcoach.chat.agent import ask
from kickcoach.chat.tools import Tools
from kickcoach.llm import GeminiClient
from kickcoach.llm.gemini import DEV_MODEL
from kickcoach.storage import db

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--player", default="me")
    ap.add_argument("--db", type=Path, default=db.DB_PATH)
    ap.add_argument("--show-tools", action="store_true", help="print each tool call")
    args = ap.parse_args()
    con = db.connect(args.db)
    tools = Tools(con, args.player)
    client = GeminiClient(DEV_MODEL, ROOT / "data" / "cache" / "chat")
    history: list[dict] = []
    while True:
        q = args.question or input("you> ").strip()
        if not q or q in {"quit", "exit"}:
            break
        out = ask(q, tools, client, history=history)
        if args.show_tools:
            for c in out["tool_calls"]:
                print(f"  [tool] {c['name']}({c['args']})")
        print(f"coach> {out['answer']}")
        if out["fallback_reason"]:
            print(f"  ({out['fallback_reason']})")
        history += [{"role": "user", "text": q}, {"role": "model", "text": out["answer"]}]
        if args.question:
            break
    con.close()


if __name__ == "__main__":
    main()
