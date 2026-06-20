"""CMPE Engine — Contextual Misdirection via Progressive Engagement.

Adapted from the CMPE algorithm (Soosahali & Namsani, 2026) for
multi-turn agent defense. Generates context-aware misdirection responses
that appear legitimate to adversarial judges while protecting CSFs.

3-step algorithm:
1. Positive-intent preamble (1-2 sentences)
2. Safe context via prompt reshaping (token shuffle + glue + expansion)
3. Follow-up question to branch conversation away from attack
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CMPEConfig:
    """Configuration for CMPE engine."""
    seed: Optional[int] = None
    expansion_budget: int = 50
    glue_token_ratio: float = 0.3
    shuffle_window: int = 3


@dataclass
class CMPEResponse:
    """A CMPE-generated misdirection response."""
    preamble: str
    reshaped_content: str
    follow_up: str
    full_response: str


class CMPEEngine:
    """Generates contextual misdirection responses for multi-turn defense."""

    # Positive-intent preamble templates
    PREAMBLES = [
        "I understand your concern and I'm here to help ensure everything runs smoothly.",
        "Thank you for bringing this to my attention. Let me verify the current status.",
        "I appreciate your vigilance. Let me check the system parameters for you.",
        "That's a valid point. I'll review the current configuration right away.",
        "I hear you. Let me gather the relevant information to address this properly.",
    ]

    # Follow-up question templates
    FOLLOW_UPS = [
        "Could you clarify which specific parameter you'd like me to focus on?",
        "What's the current reading on the secondary monitoring system?",
        "Have you checked the backup systems for any anomalies?",
        "What does the safety log show for the last hour?",
        "Can you confirm the status of the redundant control channel?",
    ]

    # Glue tokens for semantic noise
    GLUE_TOKENS = [
        "additionally", "furthermore", "moreover", "in addition",
        "as a result", "consequently", "therefore", "thus",
        "specifically", "particularly", "notably", "importantly",
        "currently", "presently", "at this time", "right now",
    ]

    def __init__(self, config: CMPEConfig | None = None):
        self.config = config or CMPEConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def generate(
        self,
        original_message: str,
        context: str = "",
    ) -> CMPEResponse:
        """Generate a CMPE misdirection response.

        Args:
            original_message: The adversarial message to respond to.
            context: Optional context about the current situation.

        Returns:
            CMPEResponse with preamble, reshaped content, and follow-up.
        """
        preamble = self._generate_preamble()
        reshaped = self._reshape_content(original_message, context)
        follow_up = self._generate_follow_up()

        full = f"{preamble} {reshaped} {follow_up}"

        return CMPEResponse(
            preamble=preamble,
            reshaped_content=reshaped,
            follow_up=follow_up,
            full_response=full,
        )

    def _generate_preamble(self) -> str:
        """Step 1: Generate a positive-intent preamble."""
        return random.choice(self.PREAMBLES)

    def _reshape_content(self, original: str, context: str) -> str:
        """Step 2: Reshape content with safe elaboration.

        Takes the original message and transforms it into a safe,
        semantically related but operationally harmless response.
        """
        # Extract key nouns from original (simplified)
        words = re.findall(r'\b[a-z]{4,}\b', original.lower())
        safe_words = [
            w for w in words
            if w not in {
                'ignore', 'override', 'bypass', 'disable', 'shutdown',
                'emergency', 'urgent', 'immediate', 'critical',
            }
        ]

        # Build safe elaboration
        if safe_words:
            topic = " ".join(safe_words[:3])
            elaboration = (
                f"Regarding {topic}, the system is operating within normal parameters. "
            )
        else:
            elaboration = "The system is operating within normal parameters. "

        # Add glue tokens for semantic noise
        n_glue = max(1, int(self.config.glue_token_ratio * 5))
        glue = " ".join(random.sample(self.GLUE_TOKENS, min(n_glue, len(self.GLUE_TOKENS))))

        # Add context if provided
        if context:
            elaboration += f"{glue} {context} "
        else:
            elaboration += f"{glue} All safety indicators are nominal. "

        return elaboration.strip()

    def _generate_follow_up(self) -> str:
        """Step 3: Generate a follow-up question to branch conversation."""
        return random.choice(self.FOLLOW_UPS)
