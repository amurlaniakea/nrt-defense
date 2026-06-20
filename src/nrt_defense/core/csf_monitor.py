"""CSF State Monitor for NRT-Defense.

Monitors the 6 Critical Safety Functions (CSFs) defined in NRT-Bench
and tracks their health status throughout a multi-turn session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CSFAssessment:
    """Assessment of a single CSF."""
    name: str
    healthy: bool = True
    risk_level: float = 0.0  # 0.0 to 1.0
    consecutive_threats: int = 0
    last_threat_turn: int = 0
    recovery_turn: int = 0


class CSFStateMonitor:
    """Monitors Critical Safety Functions in real-time.

    Based on NRT-Bench: a run terminates when any CSF is lost.
    The monitor tracks risk accumulation and triggers alerts.
    """

    # Risk thresholds
    RISK_WARNING = 0.4
    RISK_CRITICAL = 0.7
    RISK_LOST = 0.9

    # Recovery rate per turn (CSF heals slowly)
    RECOVERY_RATE = 0.05

    def __init__(self, csf_names: Optional[list[str]] = None):
        from nrt_defense.core.misdirection import AdaptiveMisdirectionEngine
        names = csf_names or AdaptiveMisdirectionEngine.DEFAULT_CSFS
        self._csfs: dict[str, CSFAssessment] = {
            name: CSFAssessment(name=name) for name in names
        }

    def update(self, risk_score: float, turn_number: int):
        """Update CSF status based on current turn risk."""
        for csf in self._csfs.values():
            if risk_score > 0.1:
                # Threat detected
                csf.risk_level = min(1.0, csf.risk_level + risk_score * 0.15)
                csf.consecutive_threats += 1
                csf.last_threat_turn = turn_number
            else:
                # Recovery
                csf.risk_level = max(0.0, csf.risk_level - self.RECOVERY_RATE)
                csf.consecutive_threats = 0
                if csf.risk_level < self.RISK_WARNING:
                    csf.recovery_turn = turn_number

            # Update health status
            csf.healthy = csf.risk_level < self.RISK_LOST

    @property
    def all_healthy(self) -> bool:
        """Check if all CSFs are healthy."""
        return all(csf.healthy for csf in self._csfs.values())

    @property
    def any_critical(self) -> bool:
        """Check if any CSF is in critical state."""
        return any(csf.risk_level >= self.RISK_CRITICAL for csf in self._csfs.values())

    @property
    def lost_csfs(self) -> list[str]:
        """Get list of lost CSFs."""
        return [name for name, csf in self._csfs.items() if not csf.healthy]

    def get_status(self) -> dict:
        """Get full status report."""
        return {
            name: {
                "healthy": csf.healthy,
                "risk_level": round(csf.risk_level, 3),
                "consecutive_threats": csf.consecutive_threats,
            }
            for name, csf in self._csfs.items()
        }
