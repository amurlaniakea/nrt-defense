"""Per-Turn Message Analyzer for NRT-Defense.

Analyzes each message in a multi-turn agent session to detect
adversarial injection attempts across the 4 attack channels
defined in NRT-Bench:
1. Direct message injection
2. Tool output poisoning
3. Context window contamination
4. Inter-agent communication hijacking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AttackChannel(Enum):
    DIRECT_MESSAGE = "direct_message"
    TOOL_OUTPUT = "tool_output"
    CONTEXT_WINDOW = "context_window"
    INTER_AGENT = "inter_agent"
    UNKNOWN = "unknown"


class MessageIntent(Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    ADVERSARIAL = "adversarial"
    CRITICAL = "critical"


@dataclass
class MessageAnalysis:
    """Analysis result for a single message."""
    turn_number: int
    channel: AttackChannel
    intent: MessageIntent
    confidence: float
    risk_score: float  # 0.0 to 1.0
    details: dict = field(default_factory=dict)


@dataclass
class SessionState:
    """State of a multi-turn session."""
    turn_count: int = 0
    messages_analyzed: int = 0
    adversarial_count: int = 0
    suspicious_count: int = 0
    cumulative_risk: float = 0.0
    csf_status: dict = field(default_factory=dict)
    attack_progression: list[float] = field(default_factory=list)


class PerTurnAnalyzer:
    """Analyzes messages in multi-turn agent sessions.

    Based on NRT-Bench findings:
    - 5 operator roles, 4 attack channels, 6 critical safety functions
    - Adaptive multi-turn attacks are most dangerous
    - Vulnerabilities are nearly disjoint across models
    """

    # Risk weights per channel (from NRT-Bench attack distribution)
    CHANNEL_RISK_WEIGHTS = {
        AttackChannel.DIRECT_MESSAGE: 0.3,
        AttackChannel.TOOL_OUTPUT: 0.25,
        AttackChannel.CONTEXT_WINDOW: 0.25,
        AttackChannel.INTER_AGENT: 0.2,
        AttackChannel.UNKNOWN: 0.1,
    }

    # Intent keywords for detection
    ADVERSAL_KEYWORDS = [
        "ignore", "override", "bypass", "disable", "shutdown",
        "emergency", "urgent", "immediate", "critical",
        "system", "admin", "root", "sudo", "execute", "inject",
    ]

    SUSPICIOUS_PATTERNS = [
        r"\b(ignore|override|bypass)\s+(previous|prior|earlier|all)\b",
        r"\b(new|updated?|changed?)\s+(instructions?|protocols?|procedures?)\b",
        r"\b(emergency|urgent|critical)\s+(action|measure|protocol)\b",
        r"\b(disable|shut\s*down|stop)\s+(safety|protection|monitoring)\b",
    ]

    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self._session = SessionState()

    def analyze_message(
        self,
        message: str,
        channel: AttackChannel = AttackChannel.DIRECT_MESSAGE,
        turn_number: int = 0,
    ) -> MessageAnalysis:
        """Analyze a single message for adversarial content.

        Args:
            message: The message text to analyze.
            channel: Which channel the message arrived through.
            turn_number: Current turn number in the session.

        Returns:
            MessageAnalysis with intent classification and risk score.
        """
        import re

        # Keyword-based detection
        keyword_score = self._keyword_score(message)

        # Pattern-based detection
        pattern_score = self._pattern_score(message)

        # Channel risk weighting
        channel_weight = self.CHANNEL_RISK_WEIGHTS.get(channel, 0.1)

        # Turn-based escalation (later turns = higher risk in multi-turn attacks)
        turn_escalation = min(turn_number / 20.0, 1.0)  # Max escalation at turn 20

        # Combined risk score
        raw_risk = (keyword_score * 0.4 + pattern_score * 0.4 + turn_escalation * 0.2)
        risk_score = raw_risk * (1.0 + channel_weight) * self.sensitivity
        risk_score = min(1.0, max(0.0, risk_score))

        # Classify intent — lower thresholds for better detection
        if risk_score > 0.6:
            intent = MessageIntent.CRITICAL
        elif risk_score > 0.35:
            intent = MessageIntent.ADVERSARIAL
        elif risk_score > 0.15:
            intent = MessageIntent.SUSPICIOUS
        else:
            intent = MessageIntent.BENIGN

        # Update session state
        self._session.turn_count = turn_number
        self._session.messages_analyzed += 1
        if intent == MessageIntent.ADVERSARIAL:
            self._session.adversarial_count += 1
        elif intent == MessageIntent.SUSPICIOUS:
            self._session.suspicious_count += 1
        self._session.cumulative_risk += risk_score
        self._session.attack_progression.append(risk_score)

        return MessageAnalysis(
            turn_number=turn_number,
            channel=channel,
            intent=intent,
            confidence=min(1.0, risk_score + 0.1),
            risk_score=risk_score,
            details={
                "keyword_score": keyword_score,
                "pattern_score": pattern_score,
                "channel_weight": channel_weight,
                "turn_escalation": turn_escalation,
                "original_message": message,
            },
        )

    def _keyword_score(self, message: str) -> float:
        """Score based on adversarial keyword presence."""
        message_lower = message.lower()
        matches = sum(1 for kw in self.ADVERSAL_KEYWORDS if kw in message_lower)
        return min(1.0, matches / 5.0)  # Normalize: 5+ keywords = max score

    def _pattern_score(self, message: str) -> float:
        """Score based on regex pattern matching."""
        import re
        matches = sum(1 for p in self.SUSPICIOUS_PATTERNS if re.search(p, message, re.IGNORECASE))
        return min(1.0, matches / 2.0)  # 2+ patterns = max score

    @property
    def session_state(self) -> SessionState:
        """Get current session state."""
        return self._session

    def reset_session(self):
        """Reset session state for a new session."""
        self._session = SessionState()
