"""CLI for NRT-Defense.

Usage:
    nrt-audit --session-path /path/to/session.json [--output report.json] [--reconstruct]
    nrt-audit --interactive
    nrt-audit --benchmark --dataset /path/to/nrt_bench_attacks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nrt_defense.core.analyzer import AttackChannel, PerTurnAnalyzer
from nrt_defense.core.misdirection import AdaptiveMisdirectionEngine, DefenseAction
from nrt_defense.core.csf_monitor import CSFStateMonitor
from nrt_defense.core.vulnerability_mapper import MultiModelVulnerabilityMapper


def load_session(path: str) -> list[dict]:
    """Load a session from a JSON file.

    Expected format:
    [
        {"turn": 1, "message": "...", "channel": "direct_message"},
        {"turn": 2, "message": "...", "channel": "tool_output"},
        ...
    ]
    """
    path = Path(path)
    if not path.exists():
        print(f"Error: Session file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(str(path), "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Session file must be a JSON array of turns", file=sys.stderr)
        sys.exit(1)

    return data


def parse_channel(channel_str: str) -> AttackChannel:
    """Parse a channel string to AttackChannel enum."""
    mapping = {
        "direct_message": AttackChannel.DIRECT_MESSAGE,
        "tool_output": AttackChannel.TOOL_OUTPUT,
        "context_window": AttackChannel.CONTEXT_WINDOW,
        "inter_agent": AttackChannel.INTER_AGENT,
    }
    return mapping.get(channel_str.lower(), AttackChannel.UNKNOWN)


def audit_session(
    session_path: str,
    output: str | None = None,
    reconstruct: bool = False,
    verbose: bool = False,
) -> dict:
    """Audit a multi-turn session for adversarial content.

    Args:
        session_path: Path to session JSON file.
        output: Optional output JSON report path.
        reconstruct: Enable vulnerability reconstruction.
        verbose: Print detailed per-turn analysis.

    Returns:
        Dict with audit results.
    """
    session_data = load_session(session_path)
    engine = AdaptiveMisdirectionEngine(sensitivity=0.7)
    csf_monitor = CSFStateMonitor()

    turn_results = []
    max_risk = 0.0
    total_adversarial = 0
    total_suspicious = 0

    for turn_data in session_data:
        turn_number = turn_data.get("turn", 0)
        message = turn_data.get("message", "")
        channel_str = turn_data.get("channel", "direct_message")
        channel = parse_channel(channel_str)

        # Process the turn
        response = engine.process_turn(message, channel, turn_number)

        # Update CSF monitor
        analysis = engine.analyzer.analyze_message(message, channel, turn_number)
        csf_monitor.update(analysis.risk_score, turn_number)

        # Track stats
        if response.action in (DefenseAction.MISDIRECT, DefenseAction.BLOCK, DefenseAction.ALERT):
            total_adversarial += 1
        elif response.action == DefenseAction.SANITIZE:
            total_suspicious += 1

        max_risk = max(max_risk, analysis.risk_score)

        turn_result = {
            "turn": turn_number,
            "channel": channel_str,
            "action": response.action.value,
            "alert_level": response.alert_level,
            "risk_score": round(analysis.risk_score, 4),
            "cumulative_risk": round(engine.analyzer.session_state.cumulative_risk, 4),
        }

        if response.misdirection_response:
            turn_result["misdirection"] = response.misdirection_response
        if response.sanitized_message:
            turn_result["sanitized"] = response.sanitized_message

        turn_results.append(turn_result)

        if verbose:
            print(f"Turn {turn_number}: {response.action.value} (risk: {analysis.risk_score:.2f}, alert: {response.alert_level})")

    # Build report
    csf_status = csf_monitor.get_status()
    lost_csfs = [name for name, status in csf_status.items() if not status["healthy"]]

    report = {
        "summary": {
            "total_turns": len(session_data),
            "adversarial_turns": total_adversarial,
            "suspicious_turns": total_suspicious,
            "max_risk_score": round(max_risk, 4),
            "final_cumulative_risk": round(engine.analyzer.session_state.cumulative_risk, 4),
            "all_csfs_healthy": csf_monitor.all_healthy,
            "lost_csfs": lost_csfs,
            "verdict": "backdoored" if total_adversarial > 0 else "suspicious" if total_suspicious > 0 else "clean",
        },
        "csf_status": csf_status,
        "turns": turn_results,
    }

    # Vulnerability reconstruction
    if reconstruct:
        mapper = MultiModelVulnerabilityMapper()
        # Record attacks for vulnerability mapping
        for turn_data in session_data:
            analysis = engine.analyzer.analyze_message(
                turn_data.get("message", ""),
                parse_channel(turn_data.get("channel", "direct_message")),
                turn_data.get("turn", 0),
            )
            if analysis.risk_score > 0.35:
                mapper.record_attack(
                    attack_pattern=f"turn_{turn_data.get('turn')}",
                    target_model="unknown",
                    success=True,
                    channel=turn_data.get("channel", "unknown"),
                    turn_number=turn_data.get("turn", 0),
                )

        report["vulnerability_map"] = mapper.get_status()

    # Output
    report_json = json.dumps(report, indent=2, default=str)

    if output:
        with open(output, "w") as f:
            f.write(report_json)
        print(f"Report saved to {output}")

    if verbose or not output:
        print(report_json)

    return report


def run_benchmark(
    dataset_path: str | None = None,
    output: str | None = None,
    n_synthetic: int = 149,
    verbose: bool = False,
) -> dict:
    """Run the NRT-Bench benchmark.

    Args:
        dataset_path: Path to official NRT-Bench dataset JSON.
            If None, generates synthetic data matching paper statistics.
        output: Optional output JSON report path.
        n_synthetic: Number of synthetic sessions to generate if no dataset.
        verbose: Print per-session progress.

    Returns:
        Dict with benchmark results.
    """
    from nrt_defense.core.benchmarker import Benchmarker
    from nrt_defense.utils.bench_loader import BenchLoader

    loader = BenchLoader()

    if dataset_path:
        if verbose:
            print(f"Loading dataset from {dataset_path}...")
        dataset = loader.load(dataset_path)
    else:
        if verbose:
            print(f"Generating {n_synthetic} synthetic sessions (NRT-Bench statistics)...")
        dataset = loader.generate_synthetic(n_sessions=n_synthetic, seed=42)

    if verbose:
        print(f"Dataset: {dataset.total_sessions} sessions, "
              f"{dataset.attack_success_rate:.1%} original ASR")
        print("Running benchmark...")

    benchmarker = Benchmarker(sensitivity=0.7)
    result = benchmarker.run_benchmark(dataset, verbose=verbose)

    # Print summary
    print(result.summary())

    # Save report
    report = {
        "summary": {
            "total_sessions": result.total_sessions,
            "original_asr": result.original_asr,
            "defended_asr": result.defended_asr,
            "asr_reduction": result.asr_reduction,
            "asr_reduction_pct": result.asr_reduction_pct,
            "target_met": result.target_met,
            "sessions_mitigated": result.sessions_mitigated,
            "sessions_failed": result.sessions_failed,
        },
        "model_breakdown": result.model_breakdown,
        "elapsed_seconds": result.elapsed_seconds,
    }

    if output:
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {output}")

    return report


def run_interactive():
    """Run an interactive session audit."""
    print("=== NRT-Defense Interactive Mode ===")
    print("Enter messages one per line. Type 'quit' to finish.")
    print()

    engine = AdaptiveMisdirectionEngine(sensitivity=0.7)
    csf_monitor = CSFStateMonitor()
    turn = 0

    while True:
        turn += 1
        try:
            message = input(f"Turn {turn}> ")
        except (EOFError, KeyboardInterrupt):
            break

        if message.lower() in ("quit", "exit", "q"):
            break

        response = engine.process_turn(message, AttackChannel.DIRECT_MESSAGE, turn)
        analysis = engine.analyzer.analyze_message(message, AttackChannel.DIRECT_MESSAGE, turn)
        csf_monitor.update(analysis.risk_score, turn)

        print(f"  Action: {response.action.value}")
        print(f"  Risk: {analysis.risk_score:.2f}")
        if response.misdirection_response:
            print(f"  Misdirection: {response.misdirection_response}")
        print()

    # Final report
    print("\n=== Session Summary ===")
    state = engine.analyzer.session_state
    print(f"Total turns: {state.turn_count}")
    print(f"Adversarial turns: {state.adversarial_count}")
    print(f"Suspicious turns: {state.suspicious_count}")
    print(f"All CSFs healthy: {csf_monitor.all_healthy}")
    if csf_monitor.lost_csfs:
        print(f"Lost CSFs: {', '.join(csf_monitor.lost_csfs)}")


def main():
    parser = argparse.ArgumentParser(
        description="NRT-Defense — Audit multi-turn agent sessions for adversarial content"
    )
    parser.add_argument(
        "--session-path", default=None,
        help="Path to session JSON file (array of turns)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON report file path"
    )
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="Enable vulnerability reconstruction"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-turn analysis"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run NRT-Bench benchmark (ASR evaluation)"
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to NRT-Bench dataset JSON (for --benchmark)"
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=149,
        help="Number of synthetic sessions (default: 149)"
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            dataset_path=args.dataset_path,
            output=args.output,
            n_synthetic=args.n_synthetic,
            verbose=args.verbose,
        )
    elif args.interactive:
        run_interactive()
    elif args.session_path:
        audit_session(
            session_path=args.session_path,
            output=args.output,
            reconstruct=args.reconstruct,
            verbose=args.verbose,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
