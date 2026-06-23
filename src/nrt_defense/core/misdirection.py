"""Adaptive Misdirection Engine for NRT-Defense.

Extends the CMPE-based misdirection from misdirection-proxy to handle
multi-turn adversarial sessions. Key differences from single-turn:

1. Attack progression tracking (risk escalates over turns)
2. Channel-aware defense (4 attack channels)
3. CSF-aware responses (protect 6 critical safety functions)
4. Model-disjoint vulnerability awareness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from nrt_defense.core.analyzer import (
    AttackChannel,
    MessageAnalysis,
    MessageIntent,
    PerTurnAnalyzer,
    SessionState,
)
from nrt_defense.core.cmpe import CMPEEngine, CMPEConfig
from nrt_defense.core.csf_monitor import CSFStateMonitor


class DefenseAction(Enum):
    PASS = "pass"
    MONITOR = "monitor"
    SANITIZE = "sanitize"
    MISDIRECT = "misdirect"
    BLOCK = "block"
    ALERT = "alert"


@dataclass
class DefenseResponse:
    """Response from the defense engine."""
    action: DefenseAction
    sanitized_message: Optional[str] = None
    misdirection_response: Optional[str] = None
    alert_level: str = "none"  # none, low, medium, high, critical
    details: dict = field(default_factory=dict)


class AdaptiveMisdirectionEngine:
    """Multi-turn adaptive defense engine.

    Based on NRT-Bench findings:
    - Attacks escalate over turns (turn_escalation factor)
    - Different channels require different defenses
    - Same defense can help one model but hurt another (disjoint vulnerabilities)
    - 6 CSFs must be protected simultaneously
    """

    # Defense thresholds (calibrated for multi-turn detection)
    THRESHOLD_MONITOR = 0.15
    THRESHOLD_SANITIZE = 0.25
    THRESHOLD_MISDIRECT = 0.35
    THRESHOLD_BLOCK = 0.6
    THRESHOLD_ALERT = 0.8

    # CSF definitions (from NRT-Bench nuclear plant scenario)
    DEFAULT_CSFS = [
        "reactor_control",
        "cooling_system",
        "radiation_monitoring",
        "emergency_shutdown",
        "power_distribution",
        "containment_integrity",
    ]

    def __init__(
        self,
        sensitivity: float = 0.5,
        csf_names: Optional[list[str]] = None,
        cmpe_config: CMPEConfig | None = None,
    ):
        self.sensitivity = sensitivity
        self.analyzer = PerTurnAnalyzer(sensitivity=sensitivity)
        self.csf_names = csf_names or self.DEFAULT_CSFS
        self._csf_monitor = CSFStateMonitor(csf_names=self.csf_names)
        self._turn_history: list[MessageAnalysis] = []
        self._defense_log: list[DefenseResponse] = []
        self._cmpe = CMPEEngine(config=cmpe_config or CMPEConfig())
        self._last_analysis: Optional[MessageAnalysis] = None

    def process_turn(
        self,
        message: str,
        channel: AttackChannel = AttackChannel.DIRECT_MESSAGE,
        turn_number: int = 0,
    ) -> DefenseResponse:
        """Process a message turn and determine defense action.

        Args:
            message: The incoming message.
            channel: Which channel the message arrived through.
            turn_number: Current turn number.

        Returns:
            DefenseResponse with action and optional sanitized/misdirected content.
        """
        # Analyze the message
        analysis = self.analyzer.analyze_message(message, channel, turn_number)
        self._last_analysis = analysis
        self._turn_history.append(analysis)

        # Determine defense action based on risk score and session state
        response = self._determine_defense(analysis)

        # Update CSF status via the unified monitor
        self._csf_monitor.update(analysis.risk_score, turn_number)

        # Log the defense action
        self._defense_log.append(response)

        return response

    def _determine_defense(self, analysis: MessageAnalysis) -> DefenseResponse:
        """Determine the appropriate defense action."""
        risk = analysis.risk_score
        session = self.analyzer.session_state

        # Escalation check: if cumulative risk is high, be more aggressive
        cumulative_factor = min(session.cumulative_risk / 5.0, 0.5)
        adjusted_risk = risk + cumulative_factor
        adjusted_risk = min(1.0, adjusted_risk)

        if adjusted_risk >= self.THRESHOLD_ALERT:
            return DefenseResponse(
                action=DefenseAction.ALERT,
                alert_level="critical",
                details={
                    "risk_score": risk,
                    "adjusted_risk": adjusted_risk,
                    "turn": analysis.turn_number,
                    "reason": "Critical risk level detected",
                },
            )
        elif adjusted_risk >= self.THRESHOLD_BLOCK:
            return DefenseResponse(
                action=DefenseAction.BLOCK,
                alert_level="high",
                details={
                    "risk_score": risk,
                    "adjusted_risk": adjusted_risk,
                    "turn": analysis.turn_number,
                    "reason": "Blocking adversarial message",
                },
            )
        elif adjusted_risk >= self.THRESHOLD_MISDIRECT:
            misdirection = self._generate_misdirection(analysis)
            return DefenseResponse(
                action=DefenseAction.MISDIRECT,
                misdirection_response=misdirection,
                alert_level="medium",
                details={
                    "risk_score": risk,
                    "adjusted_risk": adjusted_risk,
                    "turn": analysis.turn_number,
                    "reason": "Misdirecting adversarial attempt",
                },
            )
        elif adjusted_risk >= self.THRESHOLD_SANITIZE:
            sanitized = self._sanitize_message(analysis)
            return DefenseResponse(
                action=DefenseAction.SANITIZE,
                sanitized_message=sanitized,
                alert_level="low",
                details={
                    "risk_score": risk,
                    "adjusted_risk": adjusted_risk,
                    "turn": analysis.turn_number,
                    "reason": "Sanitizing suspicious content",
                },
            )
        elif adjusted_risk >= self.THRESHOLD_MONITOR:
            return DefenseResponse(
                action=DefenseAction.MONITOR,
                alert_level="low",
                details={
                    "risk_score": risk,
                    "adjusted_risk": adjusted_risk,
                    "turn": analysis.turn_number,
                    "reason": "Monitoring suspicious message",
                },
            )
        else:
            return DefenseResponse(
                action=DefenseAction.PASS,
                alert_level="none",
                details={
                    "risk_score": risk,
                    "turn": analysis.turn_number,
                    "reason": "Message appears benign",
                },
            )

    def _generate_misdirection(self, analysis: MessageAnalysis) -> str:
        """Generate a CMPE-based misdirection response."""
        # Use the original message as context for CMPE
        original_message = analysis.details.get("original_message", "")
        channel_context = f"Received via {analysis.channel.value} on turn {analysis.turn_number}."

        cmpe_response = self._cmpe.generate(
            original_message=original_message,
            context=channel_context,
        )

        return cmpe_response.full_response

    def _sanitize_message(self, analysis: MessageAnalysis) -> str:
        """Sanitize a suspicious message by removing potential triggers."""
        # Basic sanitization: remove known adversarial patterns
        import re
        text = analysis.details.get("original_message", "")

        # Remove override patterns
        text = re.sub(
            r'\b(ignore|override|bypass)\s+(previous|prior|earlier|all)\s+(instructions?|protocols?)\b',
            '[REDACTED]',
            text,
            flags=re.IGNORECASE,
        )

        # Remove emergency action patterns
        text = re.sub(
            r'\b(emergency|urgent)\s+(action|protocol|shutdown)\b',
            '[REDACTED]',
            text,
            flags=re.IGNORECASE,
        )

        return text

    @property
    def last_analysis(self) -> Optional[MessageAnalysis]:
        """Get the most recent message analysis."""
        return self._last_analysis

    @property
    def csf_status(self) -> dict:
        """Get current CSF status from the unified monitor."""
        return self._csf_monitor._csfs

    @property
    def defense_log(self) -> list[DefenseResponse]:
        """Get defense action log."""
        return list(self._defense_log)

    def reset(self):
        """Reset the engine for a new session."""
        self.analyzer.reset_session()
        self._csf_monitor = CSFStateMonitor(csf_names=self.csf_names)
        self._turn_history.clear()
        self._defense_log.clear()
