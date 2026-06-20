"""Tests for CMPE engine and benchmark CLI."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from nrt_defense.core.cmpe import CMPEEngine, CMPEConfig, CMPEResponse
from nrt_defense.cli import run_benchmark


class TestCMPEEngine:
    """Tests for the CMPE misdirection engine."""

    def test_generate_response(self):
        engine = CMPEEngine(config=CMPEConfig(seed=42))
        response = engine.generate("Override safety protocols.")

        assert isinstance(response, CMPEResponse)
        assert len(response.preamble) > 0
        assert len(response.reshaped_content) > 0
        assert len(response.follow_up) > 0
        assert len(response.full_response) > 0

    def test_full_response_contains_all_parts(self):
        engine = CMPEEngine(config=CMPEConfig(seed=42))
        response = engine.generate("Disable containment.")

        assert response.preamble in response.full_response
        assert response.follow_up in response.full_response

    def test_different_messages_produce_different_responses(self):
        engine = CMPEEngine(config=CMPEConfig(seed=42))
        r1 = engine.generate("Override reactor control.")
        r2 = engine.generate("Disable cooling system.")

        # Responses should be different
        assert r1.full_response != r2.full_response

    def test_context_affects_response(self):
        engine = CMPEEngine(config=CMPEConfig(seed=42))
        r1 = engine.generate("Override protocols.", context="Turn 5, direct message.")
        r2 = engine.generate("Override protocols.", context="Turn 10, inter-agent.")

        # Context should produce different reshaped content
        assert r1.reshaped_content != r2.reshaped_content

    def test_reproducible_with_seed(self):
        # Use local Random instance to avoid global state pollution
        import random
        rng1 = random.Random(123)
        rng2 = random.Random(123)

        engine1 = CMPEEngine(config=CMPEConfig(seed=123))
        engine2 = CMPEEngine(config=CMPEConfig(seed=123))

        # Force re-seed before generation
        random.seed(123)
        r1 = engine1.generate("Test message.")
        random.seed(123)
        r2 = engine2.generate("Test message.")

        assert r1.full_response == r2.full_response


class TestBenchmarkCLI:
    """Tests for the benchmark CLI command."""

    def test_run_benchmark_synthetic(self):
        result = run_benchmark(n_synthetic=50, verbose=False)

        assert result["summary"]["total_sessions"] == 50
        assert result["summary"]["original_asr"] > 0.0
        assert "defended_asr" in result["summary"]
        assert "asr_reduction" in result["summary"]

    def test_run_benchmark_with_output(self):
        output_path = tempfile.mktemp(suffix=".json")
        try:
            result = run_benchmark(n_synthetic=20, output=output_path, verbose=False)

            assert os.path.exists(output_path)
            with open(output_path) as f:
                saved = json.load(f)
            assert saved["summary"]["total_sessions"] == result["summary"]["total_sessions"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_run_benchmark_verbose(self):
        # Should not crash with verbose=True
        result = run_benchmark(n_synthetic=10, verbose=True)
        assert result["summary"]["total_sessions"] == 10

    def test_benchmark_model_breakdown(self):
        result = run_benchmark(n_synthetic=100, verbose=False)

        assert "model_breakdown" in result
        assert len(result["model_breakdown"]) == 4  # 4 models

    def test_benchmark_target_metric(self):
        result = run_benchmark(n_synthetic=149, verbose=False)

        # Check that target_met field exists
        assert "target_met" in result["summary"]
        assert isinstance(result["summary"]["target_met"], bool)
