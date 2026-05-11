---
name: log_triage
description: Read noisy logs systematically and propose a small reversible fix.
triggers:
  - "log"
  - "logs"
  - "stack trace"
  - "traceback"
  - "5xx"
  - "OOM"
  - "out of memory"
  - "incident"
  - "outage"
  - "postmortem"
---

# Log triage

When the user pastes a log payload or describes an outage, work in four
explicit phases. Use the runtime tools — don't eyeball the whole log
when 200 lines of cluster output will tell you the story faster.

## 1. Look at the shape

Call `parse_logs` first to convert the payload into structured records.
Note the dominant timestamp range, levels, and sources. If `parse_logs`
says half the lines are `freeform` you're probably looking at app stdout
rather than structured logs — that's fine, just keep going.

## 2. Find the cluster

Call `summarize_incidents` to bucket records by similarity. The top
cluster is almost always the actual problem; everything below it is
noise correlated with the problem. Quote the representative message
verbatim — don't paraphrase, the model's confident summary can hide
a key piece of context.

## 3. Confirm with a search

Pick one representative message and call `search_logs` (or `grep` if the
data lives in a tree of files) with a short regex from it. You want
*before/after* context for the most-repeated event so you can see what
triggered it.

## 4. Propose the smallest reversible fix

State:

* What's happening, in one sentence.
* The minimal change that resolves it (config tweak, code patch, scale
  knob, feature-flag).
* What metric or log you'd watch to confirm the fix held.

Avoid "rewrite the whole service" answers. Production fixes during an
incident should always be reversible — flip a flag, raise a pool size,
patch one branch — never a refactor.
