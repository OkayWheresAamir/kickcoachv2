"""The coach chat (Phase 9, DEC-008, DEC-043): Gemini function calling over the tools in
kickcoach.chat.tools, with the validator in kickcoach.chat.validate on every answer.

This is the one place the app lets an LLM choose what to do next, because an open-ended
question needs different data each time. The analysis pipeline stays linear (D15).

    answer = ask("Why did my knee angle drop?", Tools(con, "sam"), client)

The loop: the model calls tools (at most MAX_TOOL_ROUNDS rounds), then answers. The answer
is checked; a rejected answer goes back once with the errors (the model may call tools
again); a second rejection, a missing key or a quota error gives a deterministic answer
built from what the tools returned.
"""

from __future__ import annotations

import json

from kickcoach.chat.tools import DECLARATIONS, Tools
from kickcoach.chat.validate import check, grounding
from kickcoach.llm import GeminiClient, NoApiKey, RateLimited

PROMPT_VERSION = "chat-v1"
# heuristic: the hardest scripted question needs 2 rounds (history, then a session); 4
# leaves room without letting a confused model loop.
MAX_TOOL_ROUNDS = 4
SYSTEM = """\
You are a football (soccer) coach's assistant. You answer one player's questions about
their own instep kicks, which a separate system measured from side-on video. You see the
measurements only through the tools.

Rules:
1. Call tools to get the data; never guess a value. If a question needs several pieces of
   data, call several tools.
2. Every number you write must be copied from a tool result (you may round an angle to
   whole degrees). Do no arithmetic: the tools give changes and comparisons ready-made.
3. If a feature was not measured, say so and give the tool's reason. Don't describe or
   judge it.
4. Knee values are interior angles between thigh and shin (180 = straight). Call them
   angles, never an amount of "bend" or "flexion"; a smaller number is a more bent knee.
5. The measurements can say what changed, not why. Report what changed, what else changed
   in the same sessions, and whether the change is bigger than the typical measurement
   error. Don't invent causes.
6. When you use a coaching passage, say it in your own words and put its ID in square
   brackets, e.g. [K-12]. Cite a study comparison by its norm_ref, e.g. [N-032]. Use only
   IDs the tools returned.
7. Questions that aren't about this player's kicks or how to improve them: say briefly
   that you can only help with their measured kicks. Call no tools for them.
8. Short, plain answers a teenager would understand. No medical advice."""
RETRY = (
    "An automatic checker rejected your answer:\n{errors}\n\nWrite a corrected answer. You "
    "may call tools again."
)
NO_KEY = "the coach chat needs a GEMINI_API_KEY; this answer is a template"


def ask(
    question: str,
    tools: Tools,
    client: GeminiClient,
    *,
    history: list[dict] | None = None,
    tag: str = "",
) -> dict:
    """Answer one question. ``history`` is earlier turns as [{"role": "user" | "model",
    "text": ...}]; their tool calls aren't replayed, so the model looks things up again.

    Returns {"answer", "generated_by" ("llm" | "template"), "validator_passed",
    "fallback_reason", "tool_calls", "attempts", "model_calls"}."""
    contents = [{"role": h["role"], "parts": [{"text": h["text"]}]} for h in history or []]
    contents.append({"role": "user", "parts": [{"text": question}]})
    out = {"answer": "", "generated_by": "template", "validator_passed": None,
           "fallback_reason": None, "tool_calls": [], "attempts": [], "model_calls": 0}  # fmt: skip
    try:
        for attempt in (1, 2):
            answer = _run(contents, tools, client, out, f"{tag} attempt {attempt}")
            errors = check(answer, grounding([c["result"] for c in out["tool_calls"]], question))
            out["attempts"].append({"attempt": attempt, "answer": answer, "errors": errors})
            if not errors:
                return out | {"answer": answer, "generated_by": "llm", "validator_passed": True}
            contents.append({"role": "user", "parts": [{"text": RETRY.format(
                errors="\n".join(f"- {e}" for e in errors))}]})  # fmt: skip
        reason = "both answers failed the checker; this answer lists what the tools returned"
    except NoApiKey:
        if not out["tool_calls"]:
            result = tools.call("get_session_features", {})
            out["tool_calls"].append({"name": "get_session_features", "args": {},
                                      "result": result, "by": "template"})  # fmt: skip
        reason = NO_KEY
    except RateLimited as e:
        reason = f"Gemini unavailable ({type(e).__name__}); this answer is a template"
    return out | {"answer": fallback_answer(out["tool_calls"], reason),
                  "validator_passed": False if out["attempts"] else None,
                  "fallback_reason": reason}  # fmt: skip


def _run(contents: list[dict], tools: Tools, client: GeminiClient, out: dict, tag: str) -> str:
    for round_ in range(MAX_TOOL_ROUNDS + 1):
        last = round_ == MAX_TOOL_ROUNDS
        res = client.generate_turn(
            system=SYSTEM, contents=contents, tools=DECLARATIONS,
            tool_mode="NONE" if last else None, tag=f"{PROMPT_VERSION} {tag} turn {round_ + 1}",
        )  # fmt: skip
        out["model_calls"] += 1
        contents.append(res["content"])
        if not res["function_calls"]:
            return res["text"]
        parts = []
        for fc in res["function_calls"]:
            result = tools.call(fc["name"], fc["args"])
            out["tool_calls"].append({"name": fc["name"], "args": fc["args"], "result": result})
            parts.append({"function_response": {"name": fc["name"], "response": {
                "error": result["error"]} if "error" in result else {"output": result},
                **({"id": fc["id"]} if fc.get("id") else {})}})  # fmt: skip
        contents.append({"role": "user", "parts": parts})
    return ""  # unreachable: the last round forbids tool calls


# -- the deterministic answer ---------------------------------------------------------------
def fallback_answer(calls: list[dict], reason: str) -> str:
    lines = [f"I couldn't write a checked answer ({reason}). What I looked up:"]
    for c in calls:
        lines += [f"- {line}" for line in summarize(c["name"], c["result"])]
    if not calls:
        lines.append("- nothing yet.")
    return "\n".join(lines)


def _val(v, unit) -> str:
    return f"{v} {unit or ''}".strip()


def summarize(name: str, r: dict) -> list[str]:
    """Plain lines restating a tool result. No wording beyond the result's own fields."""
    if "error" in r:
        return [f"{name}: {r['error']}"]
    if name == "get_session_features":
        head = f"Session {r['session_id']} ({r['date']})"
        if not r["accepted"]:
            return [f"{head}: refused: {'; '.join(r['refused_because'])}"]
        out = [f"{head}:"]
        for f in r["features"]:
            out.append(f"{f['id']}: {_val(f['value'], f.get('unit'))}" if f["status"] ==
                       "measured" else f"{f['id']}: not measured ({f['reason']})")  # fmt: skip
        out += [f"fault: {f['name']} ({f['fault_id']})" for f in r["faults"]]
        return out
    if name == "get_history":
        out = [f"{r['sessions_stored']} stored sessions."]
        t = r.get("trend")
        if t:
            pts = ", ".join(str(p.get("value", "not measured")) for p in t["points_oldest_first"])
            out.append(f"{t['feature_id']}, oldest first: {pts} {t['unit'] or ''}".strip())
            if "change_first_to_last" in t:
                out.append(f"change first to last: {t['change_first_to_last']:+g}")
        return out
    if name == "compare_to_norms":
        if "comparison" not in r:
            return [f"{r['feature_id']}: {r.get('reason') or r.get('note')}"]
        c = r["comparison"]
        return [
            f"{r['feature_id']} = {_val(r['value'], r['unit'])}: {c['verdict']}"
            + (f" ({c['norm_ref']}: {c['source']})" if c.get("norm_ref") else "")
        ]
    if name == "search_drills":
        return [f"[{p['id']}] {p['source']}: {p['text']}" for p in r["passages"]]
    return [json.dumps(r)[:300]]
