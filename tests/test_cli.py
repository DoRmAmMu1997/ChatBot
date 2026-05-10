import subprocess
import sys

import pytest

pytest.importorskip("torch")


def run_help(script):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout.lower()


def test_cli_help_entrypoints():
    for script in ["train_llm.py", "train_tokenizer.py", "train_10b.py", "chatbot.py"]:
        run_help(script)


def test_tiny_cpu_smoke_train(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "train_llm.py",
            "--dataset",
            "cornell",
            "--max-pairs",
            "8",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--config",
            "configs/chatbot-tiny.yaml",
            "--max-vocab-size",
            "128",
            "--cpu",
            "--output-dir",
            str(tmp_path),
            "--checkpoint-name",
            "tiny.pt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "valid ppl" in result.stdout
    assert (tmp_path / "tiny.pt").exists()
