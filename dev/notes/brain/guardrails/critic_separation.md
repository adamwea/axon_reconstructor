# Critic separation — verifier subagent discipline

Addresses the brain-theory "monitor / error detection — STRUCTURALLY SEPARATE from actor" requirement. The actor (the loop's main conversation) is structurally bad at grading its own work: same prompt, same context, same biases → rationalization. The fix is a separate Explore subagent invocation with a narrow scope, doing pass/fail verification against external ground truth.

**Reading rule for the loop**: at the END of every code-shipping slice's actor work — BEFORE committing — spawn a critic subagent per the spec below. Only commit if the critic returns pass (or if the critic returns concerns and the user OKs proceeding).

---

## When critic separation is required

**REQUIRED** for slices that:
- Change any file under `src/`
- Add or modify tests (since the loop authoring its own test gates is the actor-grading-itself loop)
- Produce a new pipeline output that downstream code consumes (slice contracts changes)
- Refine an existing phase whose output a trusted output anchors

**NOT REQUIRED** for slices that:
- Edit `dev/notes/` files only (no executable surface)
- Edit `pyproject.toml` for non-functional changes (docs, classifiers, extra-name only)
- Pure doc additions (READMEs, comments, plan refinements)
- Auto-generated commit_log entries

When in doubt: REQUIRED. The cost of an extra Explore call is small; the cost of a self-graded bad commit is large.

---

## What the critic gets (narrow scope)

The critic subagent is invoked via the `Agent` tool with `subagent_type: Explore` (read-only). Its prompt provides ONLY:

1. **The diff** the actor proposes to commit (output of `git diff --cached` if staged, or `git diff` otherwise).
2. **The relevant trusted-output fixture(s)** the slice's surface area touches — read from `brain/trusted_outputs.md` for the affected junction (J1-J13 per dependency_graph) or TR-xxx entries.
3. **The invariants for that junction** — derived from TR-000's structural rules + any TR-xxx-specific count invariants.
4. **The slice's stated intent** — one paragraph from the actor describing what the slice is supposed to do.

The critic does NOT get:
- The actor's reasoning trail
- The broader plan context
- Other unrelated files
- The actor's tests (since those may be biased)

This narrow scoping is the inhibition module from the brain theory — the critic's "attention" is selected to the right surface, and irrelevant context is pruned.

---

## What the critic returns

A short structured response:

```
VERDICT: pass | fail | concerns
SUMMARY: <one sentence>
EVIDENCE:
  - <invariant 1>: <met | unmet | not-applicable> — <one-line>
  - <invariant 2>: <...>
  - ...
CONCERNS (if any):
  - <concern 1>: <one-line>
  - ...
```

- **pass** = all relevant invariants met; commit allowed.
- **fail** = at least one hard invariant unmet; commit BLOCKED.
- **concerns** = soft invariants partially met OR judgment calls; commit allowed but user reviews before downstream slices proceed.

The critic does NOT propose fixes. Its sole job is verification, not design. If the critic returns fail, the ACTOR (not the critic) decides whether to retry or escalate per `brain/escalation.md`.

---

## Counter against rationalization

If the critic returns fail and the actor retries → critic fails again → actor retries again, this hits **Rule 5 of `brain/escalation.md`** (critic-rejection limit, N=3 per slice). At N=3 the loop concedes the design is wrong, not the implementation, and escalates to the user.

This is the structural guard against the actor weakening its commit until the critic gives up — the critic doesn't soften over iterations; if it keeps rejecting, the design is wrong.

---

## When the critic is itself wrong

The critic can hallucinate, misread the diff, or apply invariants incorrectly. The actor can push back ONCE (with explicit reasoning in the actor's response) before requiring user adjudication. Pattern:

1. Critic: fail.
2. Actor: "Critic returned fail citing invariant X, but I believe X is not applicable here because Y. Proposing to commit anyway with this caveat." Surfaces as `concerns` to the user, NOT a silent commit.
3. User picks: (a) accept actor's reasoning, commit; (b) accept critic's verdict, revise; (c) ask for more detail.

The push-back is structural, not implicit — every disagreement surfaces to the user explicitly rather than the actor over-riding the critic silently.

---

## Mode-mode interaction

- In **`extended_autonomous`** mode: critic separation is mandatory per the rules above; loop runs it autonomously for every code-shipping slice.
- In **`collaborative`** mode: code shipping is forbidden by mode (per `brain/mode.md`), so critic separation typically doesn't trigger. EXCEPTION: if collaborative mode allows a user-picked answer that ships code (rare), critic separation still applies.
- In **`paused`** mode: code shipping is forbidden, so critic separation doesn't trigger.

---

## Implementation note

The Agent / Explore subagent invocation is what makes "separate" structurally real — different prompt, different context window, different model invocation. Not just "the actor pretends to be a critic for one paragraph." If the loop ever skips the actual Agent call and self-grades, that's the failure mode this guardrail exists to prevent.
