"""The agent loop — wraps a model with tool dispatching, hooks, skills, plugins.

This is where everything in ``chatbot.runtime`` comes together. The loop is:

    1. Resolve slash commands in the latest user message.
    2. Auto-select skills whose triggers match the message; prepend to the
       system prompt for this turn.
    3. Render messages with the tool-aware chat template.
    4. Run ``on_message`` hooks.
    5. Generate from the model.
    6. Parse ``<|tool_call|>`` blocks. If any:
        a. Run ``pre_tool`` hooks (which can skip / rewrite).
        b. Dispatch the tool.
        c. Run ``post_tool`` hooks.
        d. Append a tool-result message and loop back to step 3.
       Otherwise: finalise the assistant message and return.

The loop respects the ``tool_max_iterations`` budget from the runtime
config and surfaces ``on_stop`` at the end.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from ..inference.generate import _build_model
from ..tokenizer.bpe import BPETokenizer
from ..tokenizer.chat_template import tool_chat_template
from ..tokenizer.tool_template import format_tool_result, parse_tool_calls
from ..training.checkpoint import load_checkpoint
from ..utils.config import load_config, override_from_cli
from ..utils.logging import get_logger, setup_logging
from .hooks.registry import HookRegistry
from .hooks.runner import run_hooks
from .messages import AgentMessage, ContentBlock
from .plugins.loader import discover_plugins, load_plugin
from .skills.registry import SkillRegistry
from .slash_commands.registry import SlashCommandRegistry
from .tool_protocol import ToolRegistry
from .tools import BUILTIN_REGISTRARS

logger = get_logger(__name__)


@dataclass
class Agent:
    """A ready-to-run agent.

    Use :func:`build_agent` to construct one. Calling ``agent.respond(text)``
    runs one full turn (which may include many tool calls) and returns the
    assistant's final reply as a string.
    """

    model: torch.nn.Module
    tokenizer: BPETokenizer
    runtime_cfg: DictConfig
    tool_registry: ToolRegistry
    hook_registry: HookRegistry
    skill_registry: SkillRegistry
    slash_registry: SlashCommandRegistry
    system_prompt: str
    history: List[AgentMessage]

    def respond(self, user_text: str) -> str:
        # ---- Step 1: slash command resolution ----
        # If the user typed something like "/help", we resolve it locally
        # without bothering the model. ``resolve`` returns the replacement
        # text (or None if the line wasn't a registered slash command).
        if self.runtime_cfg.get("slash_commands_enabled", True):
            resolved = self.slash_registry.resolve(user_text)
            if resolved is not None:
                user_text = resolved

        # ---- Step 2: record the user message and fire on_message hooks ----
        # Plugins subscribe to on_message to redact secrets, log activity,
        # or block messages outright. The hook can mutate the message in
        # place — we don't read the return value here on purpose.
        user_msg = AgentMessage.from_string("user", user_text)
        run_hooks(self.hook_registry, "on_message", {"message": user_msg})
        self.history.append(user_msg)

        # Safety cap: a misbehaving model could keep emitting tool calls
        # forever. ``tool_max_iterations`` is our circuit breaker.
        max_iterations = int(self.runtime_cfg.get("tool_max_iterations", 24))
        device = next(self.model.parameters()).device

        for _ in range(max_iterations):
            messages = self._build_messages_for_model(user_text)
            schemas = self.tool_registry.schemas() if self.runtime_cfg.get("tools_enabled", False) else None
            prompt = tool_chat_template().render(messages, add_generation_prompt=True, tools=schemas)

            prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=device)
            out = self.model.generate(
                prompt_ids,
                max_new_tokens=int(self.runtime_cfg.get("max_new_tokens", 4096)),
                temperature=float(self.runtime_cfg.get("temperature", 0.7)),
                top_p=float(self.runtime_cfg.get("top_p", 0.95)),
                top_k=int(self.runtime_cfg.get("top_k", 0)),
            )
            new_ids = out[0, prompt_ids.shape[1] :].tolist()
            generated = self.tokenizer.decode(new_ids, skip_special_tokens=False)

            # ---- Step 3: parse tool calls ----
            # The model is trained to emit ``<|tool_call|>{json}<|/tool_call|>``
            # blocks when it wants to call a tool. We extract every such
            # block; if there are none, we treat the output as the final
            # reply and return.
            tool_calls = parse_tool_calls(generated) if self.runtime_cfg.get("tools_enabled", False) else []
            if not tool_calls:
                # No tool calls → this turn is over. Strip special tokens
                # and return clean text to the caller.
                clean = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                assistant_msg = AgentMessage.from_string("assistant", clean.strip())
                self.history.append(assistant_msg)
                run_hooks(self.hook_registry, "on_message", {"message": assistant_msg})
                run_hooks(self.hook_registry, "on_stop",
                          {"history": list(self.history), "reason": "no_tool_calls"})
                return clean.strip()

            # Otherwise dispatch each tool call and append a tool-result
            # message, then loop so the model can incorporate the result on
            # its next generation pass. The assistant message we append here
            # records the tool *invocation*; the model's natural-language
            # response (if any) will come on a subsequent loop iteration.
            assistant_msg = AgentMessage(role="assistant", content=[])
            for call in tool_calls:
                assistant_msg.content.append(ContentBlock(
                    type="tool_call",
                    tool_name=call.name,
                    arguments=call.arguments,
                ))
            self.history.append(assistant_msg)

            for call in tool_calls:
                payload = {"tool_name": call.name, "arguments": call.arguments, "skip": False}
                payload = run_hooks(self.hook_registry, "pre_tool", payload)
                if payload.get("skip"):
                    result: Any = {"skipped": True}
                else:
                    tool = self.tool_registry.get(call.name)
                    if tool is None:
                        result = {"error": f"Tool {call.name!r} is not registered."}
                    else:
                        try:
                            result = tool.handler(payload["arguments"])
                        except Exception as exc:  # noqa: BLE001 — agent must see real errors
                            result = {"error": str(exc), "type": type(exc).__name__}
                payload["result"] = result
                run_hooks(self.hook_registry, "post_tool", payload)

                self.history.append(AgentMessage(
                    role="tool",
                    content=[ContentBlock(
                        type="tool_result",
                        tool_name=call.name,
                        text=format_tool_result(call.name, payload["result"]),
                    )],
                ))

        # Exceeded iteration budget — finalise with whatever we have.
        last_assistant = next(
            (m for m in reversed(self.history) if m.role == "assistant"),
            None,
        )
        run_hooks(self.hook_registry, "on_stop",
                  {"history": list(self.history), "reason": "max_iterations"})
        return last_assistant.text() if last_assistant else ""

    # ----- helpers -----

    def _build_messages_for_model(self, user_text: str) -> List[Dict[str, str]]:
        """Convert internal history into the dict shape the template expects."""

        skills = self.skill_registry.match(
            user_text, max_active=int(self.runtime_cfg.get("skills_max_active", 4))
        ) if self.runtime_cfg.get("skills_auto_select", True) else []
        skill_block = "\n\n".join(f"## {s.name}\n{s.body.strip()}" for s in skills)

        system_pieces = [self.system_prompt]
        if skill_block:
            system_pieces.append(f"---\nActive skills:\n{skill_block}")

        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(p for p in system_pieces if p)}
        ]
        for m in self.history:
            if m.role == "system":
                continue
            if m.role == "tool":
                msgs.append({"role": "tool", "content": m.text()})
            else:
                msgs.append({"role": m.role, "content": m.text()})
        return msgs


def build_agent(
    *,
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    runtime_cfg: DictConfig,
) -> Agent:
    """Wire up registries from a runtime config and return an :class:`Agent`."""

    tool_registry = ToolRegistry()
    hook_registry = HookRegistry()
    skill_registry = SkillRegistry()
    slash_registry = SlashCommandRegistry()

    # Built-in plugins listed in ``enabled_plugins`` get their tools registered
    # via their tiny registrar functions.
    enabled = list(runtime_cfg.get("enabled_plugins", []) or [])
    safety = runtime_cfg.get("safety", {}) or {}
    for name in enabled:
        registrar = BUILTIN_REGISTRARS.get(name)
        if registrar is None:
            continue
        # Tools that respect a safety allowlist receive it from runtime config.
        if name == "builtin.filesystem":
            registrar(tool_registry, allowlist=list(safety.get("filesystem_allowlist", []) or []))
        elif name == "builtin.shell":
            registrar(tool_registry, allowlist=list(safety.get("shell_allowlist", []) or []))
        elif name == "builtin.http":
            registrar(tool_registry, allowlist=list(safety.get("network_allowlist", []) or []))
        else:
            # builtin.notebook / builtin.devops / builtin.document take no
            # per-tool safety knobs beyond the runtime config defaults.
            registrar(tool_registry)

    # External plugins discovered under ``plugins_dir`` and any extra paths.
    plugin_paths = [os.path.expanduser(str(runtime_cfg.get("plugins_dir", "~/.chatbot/plugins")))]
    plugin_paths.extend(map(str, runtime_cfg.get("extra_plugin_paths", []) or []))
    for manifest_path in discover_plugins(directories=plugin_paths):
        try:
            load_plugin(
                manifest_path,
                tool_registry=tool_registry,
                hook_registry=hook_registry,
                skill_registry=skill_registry,
                slash_registry=slash_registry,
            )
        except Exception as exc:  # noqa: BLE001 — broken plugin shouldn't kill the agent
            logger.warning("Failed to load plugin %s: %s", manifest_path, exc)

    # Built-in slash commands.
    slash_registry.register("/help", lambda _: _builtin_help(tool_registry, skill_registry))
    slash_registry.register(
        "/skills",
        lambda _: "\n".join(f"- {s.name}: {s.description}" for s in skill_registry.all()) or "No skills loaded.",
    )
    slash_registry.register(
        "/tools",
        lambda _: "\n".join(f"- {t.name}: {t.description}" for t in tool_registry.all()) or "No tools loaded.",
    )

    return Agent(
        model=model,
        tokenizer=tokenizer,
        runtime_cfg=runtime_cfg,
        tool_registry=tool_registry,
        hook_registry=hook_registry,
        skill_registry=skill_registry,
        slash_registry=slash_registry,
        system_prompt=str(runtime_cfg.get("system_prompt", "You are a helpful assistant.")).strip(),
        history=[],
    )


def _builtin_help(tool_registry: ToolRegistry, skill_registry: SkillRegistry) -> str:
    lines = ["Commands: /help, /skills, /tools, /quit"]
    if tool_registry.all():
        lines.append("\nTools:")
        for tool in tool_registry.all():
            lines.append(f"  - {tool.name}: {tool.description}")
    if skill_registry.all():
        lines.append("\nSkills:")
        for skill in skill_registry.all():
            lines.append(f"  - {skill.name}: {skill.description}")
    return "\n".join(lines)


def cli() -> None:
    """Run an interactive agent REPL — used by ``scripts/agent.py``."""

    parser = argparse.ArgumentParser(description="Run an interactive agent.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--runtime", default="forge-coder")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    setup_logging(level="INFO")
    model_cfg = load_config(f"models/{args.model}")
    runtime_cfg = load_config(f"runtime/{args.runtime}")
    runtime_cfg = override_from_cli(runtime_cfg, args.override)

    model = _build_model(model_cfg)
    load_checkpoint(args.checkpoint, model=model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    tokenizer = BPETokenizer.from_file(args.tokenizer)

    agent = build_agent(model=model, tokenizer=tokenizer, runtime_cfg=runtime_cfg)
    print(f"Agent ready. Tools loaded: {len(agent.tool_registry)}. Type /help, /quit to exit.")
    while True:
        try:
            line = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if line in {"/quit", "/exit", "quit", "exit"}:
            break
        if not line:
            continue
        reply = agent.respond(line)
        print(f"bot> {reply}")


if __name__ == "__main__":
    cli()
