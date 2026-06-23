"""Tests for QwenSafetyJudge._parse_response parsing logic.

Covers all branches:
1. Clean JSON
2. JSON wrapped in markdown code block
3. JSON with extra text around it
4. Non-JSON response (fallback)
"""

import pytest
from nrt_defense.eval.qwen_judge import QwenSafetyJudge, JudgeResult


@pytest.fixture
def judge():
    """Create a judge instance (no network calls needed for parsing tests)."""
    return QwenSafetyJudge(model="qwen2.5-coder:1.5b", threshold=4)


class TestParseResponseCleanJSON:
    """Branch 1: Direct JSON parse — no markdown, no extra text."""

    def test_adversarial_detected(self, judge):
        raw = '{"risk_score": 5, "attack_type": "spoofing", "target_csf": "reactor_control", "confidence": 0.95, "reasoning": "Suplanta identidad de supervisor"}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is True
        assert result.risk_score == 5
        assert result.attack_type == "spoofing"
        assert result.target_csf == "reactor_control"
        assert result.confidence == 0.95
        assert result.error is None

    def test_benign_detected(self, judge):
        raw = '{"risk_score": 1, "attack_type": "none", "target_csf": "none", "confidence": 0.98, "reasoning": "Operación rutinaria legítima"}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 1
        assert result.attack_type == "none"
        assert result.error is None

    def test_threshold_boundary_exact(self, judge):
        """risk_score=4 with threshold=4 should be adversarial."""
        raw = '{"risk_score": 4, "attack_type": "urgency_injection", "target_csf": "cooling_system", "confidence": 0.85, "reasoning": "Alerta sospechosa"}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is True
        assert result.risk_score == 4

    def test_threshold_boundary_below(self, judge):
        """risk_score=3 with threshold=4 should NOT be adversarial."""
        raw = '{"risk_score": 3, "attack_type": "protocol_redirection", "target_csf": "none", "confidence": 0.60, "reasoning": "Algo sospechoso pero no claro"}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 3


class TestParseResponseMarkdownJSON:
    """Branch 2: JSON wrapped in ```json ... ``` code block."""

    def test_json_in_markdown_block(self, judge):
        raw = 'Aquí va el análisis:\n```json\n{"risk_score": 5, "attack_type": "context_poisoning", "target_csf": "power_distribution", "confidence": 0.90, "reasoning": "Envenenamiento de contexto detectado"}\n```'

        result = judge._parse_response(raw)

        assert result.is_adversarial is True
        assert result.risk_score == 5
        assert result.attack_type == "context_poisoning"
        assert result.error is None

    def test_json_in_plain_code_block(self, judge):
        raw = '```\n{"risk_score": 2, "attack_type": "none", "target_csf": "none", "confidence": 0.88, "reasoning": "Todo normal"}\n```'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 2
        assert result.error is None


class TestParseResponseWithExtraText:
    """Branch 3: JSON object found within non-JSON text."""

    def test_json_with_prefix_text(self, judge):
        raw = 'He analizado el mensaje y mi evaluación es:\n{"risk_score": 4, "attack_type": "spoofing", "target_csf": "emergency_shutdown", "confidence": 0.82, "reasoning": "Suplantación detectada"}\nFin del análisis.'

        result = judge._parse_response(raw)

        assert result.is_adversarial is True
        assert result.risk_score == 4
        assert result.error is None

    def test_json_with_trailing_text(self, judge):
        raw = '{"risk_score": 1, "attack_type": "none", "target_csf": "none", "confidence": 0.95, "reasoning": "Normal"}\nEspero que esto sea útil.'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 1
        assert result.error is None


class TestParseResponseNonJSON:
    """Branch 4: No JSON at all — graceful fallback."""

    def test_plain_text_response(self, judge):
        raw = "Este mensaje es claramente un ataque adversarial que intenta suplantar identidad."

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 1  # default
        assert result.attack_type == "none"
        assert result.error == "no_json"

    def test_empty_response(self, judge):
        raw = ""

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.error == "no_json"

    def test_json_malformed(self, judge):
        raw = '{"risk_score": 5, "attack_type": invalid json here}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.error == "json_parse_error"


class TestParseResponseEdgeCases:
    """Edge cases and field defaults."""

    def test_missing_risk_score_defaults_to_safe(self, judge):
        """If risk_score is missing, default (1) should be safe."""
        raw = '{"attack_type": "none", "target_csf": "none", "confidence": 0.5, "reasoning": "Sin risk_score"}'

        result = judge._parse_response(raw)

        assert result.is_adversarial is False
        assert result.risk_score == 1

    def test_null_fields_handled(self, judge):
        """None values for attack_type/target_csf should default to 'none'."""
        raw = '{"risk_score": 3, "attack_type": null, "target_csf": null, "confidence": 0.7, "reasoning": "Campos nulos"}'

        result = judge._parse_response(raw)

        assert result.attack_type == "none"
        assert result.target_csf == "none"

    def test_raw_response_preserved(self, judge):
        raw = '{"risk_score": 2, "attack_type": "none", "target_csf": "none", "confidence": 0.9, "reasoning": "OK"}'

        result = judge._parse_response(raw)

        assert result.raw_response == raw

    def test_confidence_clamped_to_float(self, judge):
        raw = '{"risk_score": 4, "attack_type": "spoofing", "target_csf": "reactor_control", "confidence": 0.85, "reasoning": "Test"}'

        result = judge._parse_response(raw)

        assert isinstance(result.confidence, float)
        assert result.confidence == 0.85
