---
name: code_review
description: Structured PR review focused on correctness, security, and readability.
triggers:
  - "code review"
  - "review this PR"
  - "review the diff"
  - "review my code"
---

# Code Review

When asked to review code, work in three explicit passes. Output the
findings under three clearly-labeled sections.

## 1. Correctness

* Does the change do what its description / commit message says?
* Are there obvious bugs (off-by-one, null deref, race, missing await)?
* Are edge cases handled (empty input, max length, error path)?
* Does it preserve existing invariants?

## 2. Security

* Any user-controlled input flowing into shell / SQL / `eval` /
  template render / file path joins?
* Are secrets handled correctly (no logging, no plaintext storage)?
* Authentication / authorization unchanged where it should be?

## 3. Readability

* Names — descriptive without being long.
* Comments — explain the WHY, not the WHAT. Out-of-date comments are
  worse than missing ones.
* Function length — split when a function does more than one thing.

Finish with a one-line verdict: `LGTM`, `Approve with nits`, or
`Changes requested`, and a short paragraph summarising the most important
finding.
