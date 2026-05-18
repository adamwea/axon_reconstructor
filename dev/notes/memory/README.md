# Memory

Claude's cross-session working memory for the axon_recon repo. Distinct from:
- **Guardrails** (`../guardrails/`) — locked code contracts; rarely change.
- **Trackers** (`../trackers/`) — long-form backlog (tech_debt, roadmap, issues).
- **Plans** (`../plans/active/`) — implementation roadmaps for in-progress work.
- **Commit log** (`../commit_log.md`) — append-only history of `claude:` commits.

Memory is the **flexible**, **current**, **subjective** layer. It's where Claude records what's true right now, what's pending decision, and any in-flight context that helps the next session continue smoothly.

## Files

| File | Purpose | Lifetime of an entry |
|---|---|---|
| [`current_state.md`](current_state.md) | What's shipped, what's in-flight, what's queued. Latest shifter digest, latest smoke results. The "snapshot" view. | Updated continuously; entries decay as state changes. |
| [`open_questions.md`](open_questions.md) | TBD decisions awaiting user input or empirical data. | Each entry has a resolution criterion; deletes when resolved (moves to plan or tracker). |
| [`notes.md`](notes.md) | Free-form scratch — debugging trails, ideas in flight, half-formed thoughts. | Prune aggressively; this isn't an archive. |

## Update protocol

- **After every slice that changes shipped state**: update `current_state.md`. Add the new fact; delete the previous-state fact it replaced.
- **When a new question surfaces that doesn't have an owner**: add it to `open_questions.md` with a clear resolution criterion.
- **When the user explicitly asks "remember X"**: pick the right file (state vs question vs note) and add it there.
- **During heavy debugging**: drop trail-of-breadcrumbs notes in `notes.md` so the next session can pick up cold.

## Pruning protocol

Memory is supposed to stay short and high-signal. If a file grows past ~150 lines, prune it:
- Things that are now shipped → move to `commit_log.md`'s historical record (or just delete).
- Things that became contracts → move to the relevant `guardrails/<topic>.md`.
- Things that became plans → move to `plans/active/<plan>.md` or `trackers/<tracker>.md`.
- Things that are genuinely no-longer-relevant → delete.

The goal is "Claude reads memory in 30 seconds at session start and has the current picture". Long memory = drift = re-debug-from-scratch.

## What memory is NOT for

- **Long-form architectural reasoning** — that's a plan doc.
- **Permanent contracts** — that's a guardrail.
- **Backlog grooming** — that's a tracker.
- **History of changes** — that's `commit_log.md` and git log.
- **Project context for new collaborators** — that's the root `README.md`, `CLAUDE.md`, and the trackers.

If a note feels durable, it doesn't belong here — promote it. If it feels in-flight, it does.
