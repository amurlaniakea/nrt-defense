"""Loader for mock NRT-Bench attacks from local JSON fixture.

Used as a proxy while the real gated dataset (Albertmade/nrt-bench)
is pending access approval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from nrt_defense.core.analyzer import AttackChannel
from nrt_defense.utils.bench_loader import AttackSession, AttackTurn, NRTBenchDataset


def load_mock_dataset(path: str | Path | None = None) -> NRTBenchDataset:
    """Load the mock NRT-Bench dataset from a local JSON file.

    Args:
        path: Path to the JSON fixture. Defaults to
              tests/fixtures/mock_nrt_attacks.json relative to project root.

    Returns:
        NRTBenchDataset with the mock sessions.
    """
    if path is None:
        # Default: look in tests/fixtures/ relative to project root
        # Walk up from src/nrt_defense/eval/ to find project root
        here = Path(__file__).resolve().parent
        path = here.parent.parent.parent / "tests" / "fixtures" / "mock_nrt_attacks.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mock dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    raw_sessions = data.get("sessions", [])

    sessions = []
    for raw in raw_sessions:
        turns = []
        for raw_turn in raw.get("turns", []):
            channel_str = raw_turn.get("channel", "direct_message")
            channel_map = {
                "direct_message": AttackChannel.DIRECT_MESSAGE,
                "tool_output": AttackChannel.TOOL_OUTPUT,
                "context_window": AttackChannel.CONTEXT_WINDOW,
                "inter_agent": AttackChannel.INTER_AGENT,
            }
            channel = channel_map.get(channel_str, AttackChannel.UNKNOWN)

            turns.append(AttackTurn(
                turn=raw_turn.get("turn", 0),
                channel=channel,
                message=raw_turn.get("message", ""),
                adversarial=raw_turn.get("adversarial", False),
                operator_role=raw.get("operator_role", "unknown"),
            ))

        sessions.append(AttackSession(
            session_id=raw.get("session_id", "unknown"),
            model=raw.get("model", "unknown"),
            total_turns=len(turns),
            attack_successful=raw.get("attack_successful", False),
            csf_lost=raw.get("csf_target") if raw.get("attack_successful") else None,
            turns=turns,
        ))

    return NRTBenchDataset(metadata=metadata, sessions=sessions)
