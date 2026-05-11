"""Skill parsing tests."""

from __future__ import annotations

from chatbot.runtime.skills.loader import load_skill


def test_skill_with_frontmatter(tmp_path):
    path = tmp_path / "demo.md"
    path.write_text(
        "---\nname: demo\ndescription: a tiny skill\ntriggers: [\"review\", \"check\"]\n---\n"
        "Hello body.\n",
        encoding="utf-8",
    )
    skill = load_skill(path)
    assert skill.name == "demo"
    assert "Hello body" in skill.body
    assert "review" in skill.triggers


def test_skill_without_frontmatter_uses_filename(tmp_path):
    path = tmp_path / "barebones.md"
    path.write_text("Just a body, no metadata.", encoding="utf-8")
    skill = load_skill(path)
    assert skill.name == "barebones"
    assert "Just a body" in skill.body
