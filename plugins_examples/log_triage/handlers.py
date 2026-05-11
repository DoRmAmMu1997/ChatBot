"""Slash-command handler for /postmortem.

The handler doesn't do model inference — it just emits a templated
markdown skeleton the user (or the model) can fill in. Forge will most
often *call this directly* to get a structure, then dictate its own
content into the resulting outline.
"""

from __future__ import annotations


_TEMPLATE = """## Incident postmortem

### Summary
<one paragraph: what happened, when, who was impacted, how long>

### Timeline (UTC)
- HH:MM  First signal (alert / customer report / dashboard anomaly)
- HH:MM  Engineer paged
- HH:MM  Root cause identified
- HH:MM  Mitigation applied
- HH:MM  Incident declared resolved

### Root cause
<two or three sentences. Avoid "human error" — find the system that
allowed the error to land>

### What went well
- ...

### What went poorly
- ...

### Where we got lucky
- ...

### Action items
- [ ] Owner: ...  Due: ...
- [ ] Owner: ...  Due: ...
- [ ] Owner: ...  Due: ...
"""


def postmortem(_rest: str) -> str:
    """Return the postmortem template so the user can fill it in."""

    return _TEMPLATE
