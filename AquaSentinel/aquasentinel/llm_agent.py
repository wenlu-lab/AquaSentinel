"""Intelligent report generation via a domain-specific LLM agent.

Implements R = LLM(D, C, H, T) (Eq. 16), where
    D: detection results (RTCA reports, localization, severity)
    C: network context (topology summary, sensor layout, MoE state)
    H: historical patterns (recent alerts, baseline statistics)
    T: report template

The system prompt and the report template are NOT hardcoded; they are
loaded at runtime from the `prompts/` directory (see PromptLibrary), so
domain experts can revise reporting behavior without touching code.

Two backends are supported and selected via configuration:
  * "anthropic": Anthropic Messages API
  * "openai":    any OpenAI-compatible chat completions endpoint
Both are called through plain HTTP so no vendor SDK is required. If no
API key is configured, the agent falls back to a deterministic offline
formatter so the rest of the pipeline remains fully runnable.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------- #
# Prompt loading
# ---------------------------------------------------------------------- #
class PromptLibrary:
    """Loads prompt/skill files from a directory at runtime."""

    def __init__(self, prompt_dir: str = "prompts") -> None:
        self.dir = Path(prompt_dir)

    def load(self, name: str) -> str:
        path = self.dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}. "
                f"Place the file under '{self.dir}/'."
            )
        return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------- #
# LLM client
# ---------------------------------------------------------------------- #
@dataclass
class LLMConfig:
    backend: str = "anthropic"                  # "anthropic" | "openai"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1500
    temperature: float = 0.2
    api_key_env: str = "LLM_API_KEY"
    base_url: str = ""                          # required for "openai" backend


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env, "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.cfg.backend == "anthropic":
            return self._anthropic(system_prompt, user_prompt)
        if self.cfg.backend == "openai":
            return self._openai(system_prompt, user_prompt)
        raise ValueError(f"Unknown LLM backend: {self.cfg.backend}")

    def _post(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _anthropic(self, system_prompt: str, user_prompt: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        return "".join(
            blk.get("text", "") for blk in data.get("content", [])
            if blk.get("type") == "text"
        )

    def _openai(self, system_prompt: str, user_prompt: str) -> str:
        data = self._post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------- #
# Report agent
# ---------------------------------------------------------------------- #
@dataclass
class ReportAgent:
    llm_cfg: LLMConfig = field(default_factory=LLMConfig)
    prompt_dir: str = "prompts"
    system_prompt_file: str = "report_system.txt"
    template_file: str = "report_template.txt"

    def __post_init__(self) -> None:
        self.prompts = PromptLibrary(self.prompt_dir)
        self.client = LLMClient(self.llm_cfg)

    def generate(
        self,
        detection: Dict[str, Any],       # D: detection results
        context: Dict[str, Any],         # C: network context
        history: Optional[List[Dict[str, Any]]] = None,  # H: past patterns
    ) -> str:
        """R = LLM(D, C, H, T)."""
        system_prompt = self.prompts.load(self.system_prompt_file)
        template = self.prompts.load(self.template_file)

        user_prompt = template.format(
            detection=json.dumps(detection, indent=2, default=str),
            context=json.dumps(context, indent=2, default=str),
            history=json.dumps(history or [], indent=2, default=str),
        )

        if self.client.available:
            return self.client.complete(system_prompt, user_prompt)
        return self._offline_report(detection)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _offline_report(detection: Dict[str, Any]) -> str:
        """Deterministic fallback when no LLM endpoint is configured."""
        status = detection.get("status", "UNKNOWN")
        lines = [
            "AQUASENTINEL REPORT",
            f"Timestamp: {detection.get('timestamp', 'n/a')}",
            f"Status: {status}",
            f"Real-time anomalies: {detection.get('realtime_anomalies', 'None')}",
            f"Cumulative anomalies: {detection.get('cumulative_anomalies', 'None')}",
        ]
        for hyp in detection.get("leak_hypotheses", []):
            lines.append(f"Localization: {hyp}")
        for sev in detection.get("severity", []):
            lines.append(
                f"Node {sev['node_id']}: {sev['severity']} "
                f"(confidence {sev['confidence']:.2f}, "
                f"priority {sev['priority']:.4f})"
            )
        lines.append(
            "Action required: dispatch maintenance per priority order."
            if status != "NORMAL"
            else "Action required: continue routine monitoring."
        )
        return "\n".join(lines)
