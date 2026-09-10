# Agent Notes — noctis-swebench

## Tool-calling protocol — Qwen 3.8 only (learned 2026-09-10)

**Scope:** These rules apply only when the session model is Qwen 3.8
(e.g. `qwen3.8-flash`). Other models: ignore this section entirely — normal
prose-then-call behavior works fine for them.
- This session's harness drops tool calls when an assistant message contains
  narration text BEFORE the call block (the turn ends at the text/call boundary
  and the calls never serialize). Confirmed across repeated failures.
- When calling tools: emit the tool calls with NO preceding prose. Announce or
  summarize in a separate, normal (non-tool) message instead.
- Batch independent read-only shell calls in groups of at most 4, bare-call
  messages only.
- Never end a message that only announces intended calls ("Running batch 1:")
  — the calls must actually be present in that same message.

## Context pointers
- The latest `HANDOFF_*.md` in this root is the authoritative project state
  summary; read it before doing harness work.
- Working tree is frequently dirty with critical uncommitted changes — never
  run `git clean`, `git checkout -- .`, or `git reset --hard`.
