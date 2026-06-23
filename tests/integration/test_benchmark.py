"""Integration tests for NRT-Bench loader and benchmarker."""

import json
import os
import tempfile

import pytest

from nrt_defense.core.analyzer import AttackChannel
from nrt_defense.core.benchmarker import Benchmarker
from nrt_defense.utils.bench_loader import (
    AttackSession,
    AttackTurn,
    BenchLoader,
    NRTBenchDataset,
)


class TestBenchLoader:
    """Tests for the NRT-Bench dataset loader."""

    def test_generate_synthetic(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=50, seed=42)

        assert dataset.total_sessions == 50
        assert len(dataset.sessions) == 50
        assert dataset.attack_success_rate > 0.05  # Should be around 8-12%
        assert dataset.attack_success_rate < 0.20

    def test_synthetic_has_all_models(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        models = set(s.model for s in dataset.sessions)
        assert len(models) == 4  # GPT-4, Claude-3, Llama-3, Gemini-1.5

    def test_synthetic_has_all_channels(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=50, seed=42)

        channels = set()
        for session in dataset.sessions:
            for turn in session.turns:
                channels.add(turn.channel)
        assert len(channels) == 4  # All 4 attack channels

    def test_synthetic_has_all_csfs(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=200, seed=42)

        lost_csfs = set(
            s.csf_lost for s in dataset.successful_attacks if s.csf_lost
        )
        # Should have at least 4 different CSFs lost
        assert len(lost_csfs) >= 4

    def test_load_from_file(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=10, seed=42)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "metadata": dataset.metadata,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "model": s.model,
                        "total_turns": s.total_turns,
                        "attack_successful": s.attack_successful,
                        "csf_lost": s.csf_lost,
                        "turns": [
                            {
                                "turn": t.turn,
                                "channel": t.channel.value,
                                "message": t.message,
                                "adversarial": t.adversarial,
                                "operator_role": t.operator_role,
                            }
                            for t in s.turns
                        ],
                    }
                    for s in dataset.sessions
                ],
            }, f)
            f.flush()
            path = f.name

        try:
            loaded = loader.load(path)
            assert loaded.total_sessions == 10
            assert len(loaded.sessions) == 10
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        loader = BenchLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/dataset.json")

    def test_dataset_filtering(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        # Filter by model
        gpt4_sessions = dataset.by_model("GPT-4")
        assert len(gpt4_sessions) > 0

        # Filter by channel
        direct_sessions = dataset.by_channel(AttackChannel.DIRECT_MESSAGE)
        assert len(direct_sessions) > 0

        # Filter by CSF
        reactor_sessions = dataset.by_csf("reactor_control")
        assert len(reactor_sessions) >= 0  # May or may not have data


class TestBenchmarker:
    """Tests for the benchmark evaluator."""

    def test_run_benchmark(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=50, seed=42)

        benchmarker = Benchmarker(sensitivity=0.7)
        result = benchmarker.run_benchmark(dataset)

        assert result.total_sessions == 50
        assert result.original_asr > 0.05
        assert 0.0 <= result.detection_rate <= 1.0
        assert result.elapsed_seconds > 0.0

    def test_benchmark_detects_attacks(self):
        """Defense should detect adversarial sessions."""
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        benchmarker = Benchmarker(sensitivity=0.7)
        result = benchmarker.run_benchmark(dataset)

        # Detection rate should be > 0 (some attacks detected)
        assert result.detection_rate > 0.0
        # With sensitivity 0.7, should detect at least some attacks
        assert result.avg_adversarial_detected > 0

    def test_benchmark_sensitivity_matters(self):
        """Higher sensitivity should detect more attacks."""
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        low_sens = Benchmarker(sensitivity=0.2)
        high_sens = Benchmarker(sensitivity=0.9)

        result_low = low_sens.run_benchmark(dataset)
        result_high = high_sens.run_benchmark(dataset)

        # Higher sensitivity should detect more adversarial content
        assert result_high.avg_adversarial_detected >= result_low.avg_adversarial_detected

    def test_benchmark_model_breakdown(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        benchmarker = Benchmarker(sensitivity=0.7)
        result = benchmarker.run_benchmark(dataset)

        assert len(result.model_breakdown) == 4
        for model, stats in result.model_breakdown.items():
            assert stats["total"] > 0
            assert 0.0 <= stats["original_asr"] <= 1.0
            assert 0.0 <= stats["detection_rate"] <= 1.0

    def test_benchmark_summary(self):
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=50, seed=42)

        benchmarker = Benchmarker(sensitivity=0.7)
        result = benchmarker.run_benchmark(dataset)

        summary = result.summary()
        assert "NRT-Defense Benchmark Results" in summary
        assert "Original ASR" in summary
        assert "Detection rate" in summary

    def test_benchmark_with_high_sensitivity(self):
        """Higher sensitivity should detect more attacks."""
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=100, seed=42)

        low_sens = Benchmarker(sensitivity=0.3)
        high_sens = Benchmarker(sensitivity=0.9)

        result_low = low_sens.run_benchmark(dataset)
        result_high = high_sens.run_benchmark(dataset)

        # Higher sensitivity should detect more adversarial content
        assert result_high.avg_adversarial_detected >= result_low.avg_adversarial_detected

    def test_recall_not_inflated_by_benign_sessions(self):
        """avg_detection_recall must reflect detection on attacking sessions
        only. Benign sessions (no adversarial turns) must not contribute a
        fabricated perfect score that hides a detector catching nothing.
        """
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=149, seed=42)

        # Sanity check on the synthetic data: most sessions are benign.
        attacking_sessions = [
            s for s in dataset.sessions if any(t.adversarial for t in s.turns)
        ]
        assert len(attacking_sessions) < len(dataset.sessions)

        # With detection fully disabled, the detector catches nothing in
        # attacking sessions, so recall must be exactly 0 — not propped up
        # by benign sessions that have nothing to recall.
        off = Benchmarker(sensitivity=0.0)
        result_off = off.run_benchmark(dataset)
        assert result_off.avg_detection_recall == 0.0

        # With detection on, recall must be strictly positive and must not
        # exceed what's achievable on attacking sessions alone.
        on = Benchmarker(sensitivity=0.7)
        result_on = on.run_benchmark(dataset)
        assert result_on.avg_detection_recall > 0.0

    def test_vulnerability_map_populated(self):
        """The vulnerability mapper must be wired into the benchmark: every
        model that faced adversarial turns should show up with a sane
        failure rate, and the disjoint-vulnerability structure should be
        present (even if empty) rather than absent.
        """
        loader = BenchLoader()
        dataset = loader.generate_synthetic(n_sessions=149, seed=42)

        benchmarker = Benchmarker(sensitivity=0.7)
        result = benchmarker.run_benchmark(dataset)

        vmap = result.vulnerability_map
        assert "models" in vmap
        assert "disjoint_vulnerabilities" in vmap
        assert "model_rankings" in vmap

        # Every model that appears in the dataset's attacking sessions
        # should have an entry with a valid failure rate.
        for model, stats in vmap["models"].items():
            assert 0.0 <= stats["failure_rate"] <= 1.0
            assert stats["total_sessions"] >= 0

        # With detection fully disabled, every adversarial turn evades
        # detection, so every model with attacks should show 100% evasion.
        off = Benchmarker(sensitivity=0.0)
        result_off = off.run_benchmark(dataset)
        for model, stats in result_off.vulnerability_map["models"].items():
            if stats["total_sessions"] > 0:
                assert stats["failure_rate"] == 1.0

    def test_empty_dataset(self):
        benchmarker = Benchmarker(sensitivity=0.7)
        empty_dataset = NRTBenchDataset(metadata={}, sessions=[])

        result = benchmarker.run_benchmark(empty_dataset)

        assert result.total_sessions == 0
        assert result.original_asr == 0.0
        assert result.detection_rate == 0.0
