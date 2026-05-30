# Council Debate Config Guide

> **What is this?** The Council Debate Config controls an optional multi-round deliberation pipeline that runs *before* the Chairman writes the final answer. Instead of each model answering once and handing off to the Chairman, models answer, critique each other's responses, and then rewrite — potentially multiple times — before the Chairman synthesizes.

---

## The Default Experience (Without Debate)

Before explaining debate config, it helps to understand what happens in a **normal Full Deliberation** run with no debate rounds:

```
Your question
    ↓
Stage 1  — Each council model answers independently (parallel)
    ↓
Stage 2  — Each model anonymously ranks the other responses
    ↓
Stage 3  — Chairman synthesizes a final answer using all responses and rankings
    ↓
Stage 4  — Chairman produces a polished corrected draft
```

This is already a strong pipeline. The debate config adds **loops** between Stage 1 and Stage 3:

```
Your question
    ↓
Round 1: Stage 1 → Stage 2 → Stage 3
    ↓
Round 2: Stage 1 (rewrites) → Stage 2 (re-rank) → Stage 3 (re-synthesize)
    ↓
Round N: ...
    ↓
Stage 4 — Chairman writes final corrected draft from the last round's synthesis
```

Each rewrite round gives models a chance to incorporate feedback and sharpen their answers.

---

## Settings Reference

### Number of Rounds

Controls how many full Stage 1 → Stage 2 → Stage 3 cycles run before Stage 4.

| Rounds | What actually runs | API calls (3 models) | When to use |
|--------|-------------------|---------------------|-------------|
| **1** (default) | One-shot: S1 → S2 → S3 → S4. No debate. | ~8 | Always a safe default |
| **2** | Two cycles. Models rewrite once after seeing the Chairman's first synthesis. | ~16 | Most questions — biggest quality jump per extra cost |
| **3** | Three cycles. Rankings usually stabilize here if they're going to. | ~24 | Complex analytical questions |
| **4–5** | Maximum depth. Diminishing returns beyond round 3 in most cases. | ~32–40 | Research, adversarial fact-checking, long documents |

> **Tip:** Start with 2 rounds and Auto-Converge ON. If rankings stabilize after Round 2, the system stops automatically and you don't pay for Round 3.

---

### Critique Mode

Controls **what information each model receives at the start of Round 2+** and **how Stage 2 reviewers structure their feedback**. This only changes behaviour from Round 2 onwards — Round 1 is identical across all three modes.

---

#### Free-form (default)

**Round 1:** Every model answers your question independently with no awareness of each other.

**Stage 2:** Reviewers read all responses and rank them in natural language. No special structure required.

**Round 2+:** Each model receives:
- The Chairman's synthesis from the previous round
- Their ranking position (e.g., "you ranked 2nd of 4")

They then rewrite their response in light of that collective output.

**Example prompt a model sees at the start of Round 2:**
```
This is Round 2 of a multi-round debate on the question: "What is the best 
approach for distributed database consistency?"

Here is the Chairman's synthesis from Round 1:
[...synthesis text...]

Current rankings:
1. Response B (avg rank: 1.3)
2. Response A (avg rank: 2.1)
3. Response C (avg rank: 2.6)

Rewrite your response to be more accurate and persuasive, incorporating 
the collective insights above.
```

**Best for:** General questions, brainstorming, opinion-based queries, creative work. This is the right choice for 80% of use cases.

---

#### Paragraph-level

**Round 1:** Same as Free-form, but the backend auto-numbers every paragraph before Stage 2: `[Para 1] ... [Para 2] ...`

**Stage 2:** Reviewers must cite specific paragraph numbers in their feedback. Instead of "this section is weak," they write "Para 3 is strong; Para 5 overstates the case."

**Round 2+:** Each model receives personalized feedback:
- Which of their *own* paragraphs peers called strong or weak (with counts)
- Up to 5 top-rated paragraphs from rival models to consider incorporating

**Example of what a model sees at the start of Round 2:**
```
Round 2 — Paragraph-Level Critique

Your paragraphs from Round 1 and how peers rated them:
- [Para 1] "The CAP theorem states..." — STRONG
- [Para 2] "Therefore, eventual consistency is always preferable..." — WEAK
- [Para 3] "In practice, systems like Cassandra..." — STRONG

Top-rated paragraphs from other models you should consider:
- [Response B, Para 2] "The distinction between CP and AP systems 
  depends heavily on..." — STRONG (75% agree)
- [Response C, Para 4] "Jepsen tests have shown that..." — STRONG (100% agree)

Rewrite your response with these specific improvements in mind.
```

**Best for:** Structured essays, technical comparisons, legal or policy analysis — any topic where the response naturally has distinct sections and you want precise section-level critique rather than general impressions.

---

#### Claim-level

This mode adds an extra step *before* Stage 2 in each round.

**Before Stage 2:** The Chairman model runs an additional API call to extract "canonical claims" from all responses — specific, verifiable factual assertions — into structured JSON:

```json
{
  "Response A": [
    {"id": "A1", "claim": "PostgreSQL supports MVCC for transaction isolation"},
    {"id": "A2", "claim": "Two-phase locking is always slower than MVCC"}
  ],
  "Response B": [
    {"id": "B1", "claim": "CockroachDB uses Raft consensus for distributed transactions"},
    {"id": "B2", "claim": "Serializability is achievable without significant performance cost"}
  ]
}
```

**Stage 2:** Reviewers verdict each claim individually as `strong`, `weak`, or `neutral` — not the response as a whole. They produce a structured JSON output alongside their ranking.

**Round 2+:** Each model receives a personalized brief:
- Which of their own claims peers rejected (e.g., "Claim A2 was judged WEAK by 3/4 models")
- The top 5 claims from other models that peers strongly agreed with

**Example of what a model sees at the start of Round 2:**
```
Round 2 — Claim-Level Critique

Your claims from Round 1 and peer verdicts:
- A1: "PostgreSQL supports MVCC" — STRONG (100% agree)
- A2: "Two-phase locking is always slower than MVCC" — WEAK (75% disagree)

Top claims from other models with strong peer agreement:
- B1: "CockroachDB uses Raft consensus" — STRONG (100% agree)
- B2: "Serializability is achievable without significant cost" — STRONG (75% agree)

Rewrite your response to retract or correct weak claims and incorporate 
the strongly-agreed claims from peers where appropriate.
```

**Cost note:** This mode runs 1 extra API call (using the Chairman model) before every Stage 2. If you have 3 council members and run 3 rounds, that's 3 additional calls on top of the normal 24.

**Fallback:** If claim extraction fails or times out (90 second limit), the system automatically falls back to Free-form mode for that round. You won't lose the run — it just degrades gracefully.

**Best for:** Fact-checking, technical auditing, code review, mathematical reasoning, scientific questions — any topic where you care about the accuracy of specific claims rather than the overall narrative.

---

### Auto-Converge

When enabled, the system checks after each round whether the **aggregate ranking order has stabilized** — specifically, whether the top half of models stayed in the same relative order compared to the previous round.

If rankings are stable for N consecutive rounds (where N = the convergence threshold), the debate ends early and jumps straight to Stage 4.

**Example:** You set 5 rounds, Auto-Converge ON, threshold 2.
- After Round 2, rankings are: GPT-4.1 > Claude > Gemini
- After Round 3, rankings are: GPT-4.1 > Claude > Gemini (same)
- That's 1 stable round.
- After Round 4, rankings are: GPT-4.1 > Claude > Gemini (same again)
- That's 2 stable rounds → debate stops. Round 5 never runs.

**Turn it OFF if:** You're running a fixed research experiment and want all rounds to execute regardless of stability. Otherwise, always leave it ON.

---

### Convergence Threshold

Only relevant when Auto-Converge is ON. Controls how many consecutive stable rounds are required before stopping.

| Threshold | Meaning | Risk |
|-----------|---------|------|
| **1** | Stop after the first stable round | May stop too early — one stable round could be coincidental |
| **2** (default) | Stop after two consecutive stable rounds | Good balance |
| **3** | Stop only if three consecutive rounds are stable | Conservative; uses more API calls but high confidence in convergence |

For most use cases, leave this at **2**.

---

## Decision Guide

**Just want a better answer than a single model?**
→ Use the regular Full Deliberation mode (1 round is fine). The Chairman synthesis already combines multiple perspectives.

**Want models to challenge and refine each other?**
→ Debate ON, **2 rounds**, **Free-form**, **Auto-Converge ON**. This is the sweet spot for most questions.

**Writing a detailed technical document or essay?**
→ **2–3 rounds**, **Paragraph-level**. Paragraph citations force structured, specific critique.

**Fact-checking something where specific claims matter?**
→ **2 rounds**, **Claim-level**. Accept the extra API call cost — the structured claim verdicts are worth it.

**Doing research and want maximum thoroughness?**
→ **5 rounds**, **Free-form** (or Claim-level for factual topics), **Auto-Converge ON**, **threshold 2**.

**Testing whether the debate config works at all?**
→ **2 rounds**, **Free-form**, **Auto-Converge ON** with a simple question like *"What's the best programming language to learn in 2025?"* — fast, cheap, easy to verify the Round 2 rewrites are different from Round 1.

---

## Example: Full Claim-Level Run (2 Rounds, 3 Models)

Here's exactly what fires, in order, for a 2-round claim-level debate with 3 council models and full deliberation:

```
1.  Stage 1 (Round 1)    — 3 parallel model calls
2.  Claim extraction      — 1 Chairman call (extra, claim-level only)
3.  Stage 2 (Round 1)    — 3 parallel ranking calls (output: claim verdicts JSON)
4.  Stage 3 (Round 1)    — 1 Chairman synthesis call
5.  Check convergence     — (skipped: only 1 round completed so far)

6.  Stage 1 (Round 2)    — 3 parallel calls (each model gets personalized claim brief)
7.  Claim extraction      — 1 Chairman call (again)
8.  Stage 2 (Round 2)    — 3 parallel ranking calls
9.  Stage 3 (Round 2)    — 1 Chairman final synthesis call
10. Check convergence     — if stable, stop; otherwise continue

11. Stage 4               — 1 Chairman corrected draft call

Total: ~14 API calls (vs. ~8 for a standard single-round run)
```

---

## API / MCP Usage

If you're using the MCP tools or REST API directly, the debate settings are exposed as fields on the `council_deliberate` tool and `/api/conversations/{id}/message` endpoint:

```json
{
  "content": "What are the tradeoffs between REST and GraphQL?",
  "execution_mode": "full",
  "council_models": ["openai:gpt-4.1", "anthropic:claude-opus-4-5", "google:gemini-2.5-flash"],
  "chairman_model": "openai:gpt-4.1",
  "debate_rounds": 2,
  "critique_mode": "paragraph"
}
```

Valid values:
- `critique_mode`: `"freeform"` | `"paragraph"` | `"claim"`
- `debate_rounds`: `1` – `5`
- `execution_mode`: `"full"` | `"chat_ranking"` | `"chat_only"` (debate requires `full` or `chat_ranking`; `chat_only` forces 1 round)

The global defaults set in Settings are used when these fields are omitted.

---

## Frequently Asked Questions

**Does debate work with Chat Only mode?**
No. Chat Only (Stage 1 only) skips peer ranking entirely, so there's nothing to feed back into Round 2. The system automatically forces 1 round if you're in Chat Only mode.

**Does debate work with Chat + Ranking mode?**
Yes, but without Stage 3. Models rewrite based on peer rankings alone, with no Chairman synthesis between rounds. Stage 4 also doesn't run.

**What if a model fails during a debate round?**
The system is resilient. If one model fails Stage 1, it's excluded from that round's Stage 2 and the remaining models continue. The debate doesn't abort.

**What if claim extraction times out?**
It falls back to Free-form mode for that round automatically. You get a warning in the logs but the run continues.

**Does debate affect the Advisor system?**
No. The Council Debate Config is entirely separate from the Advisor debate system. Advisors have their own round configuration in Advisor Setup.

**Is the conversation saved with all rounds?**
Yes. The stored conversation includes the full rounds array with Stage 1/2/3 data per round. The Round Navigator in the UI lets you flip between rounds.
