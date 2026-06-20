---
title: "Securing LLM Agent Teams: Inside NRT-Defense v0.4.0"
published: false
description: "How adaptive multi-turn attacks cause critical safety function failures in 12% of agent sessions, and how to mitigate it below 1% using CMPE."
tags: python, cybersecurity, ai, opensource
canonical_url: "https://github.com/amurlaniakea/nrt-defense"
---

# Securing LLM Agent Teams: Inside NRT-Defense v0.4.0

Multi-turn autonomous LLM agents are expanding rapidly in safety-critical systems. However, a major vulnerability has been exposed by **Lee et al. (2026) in the NRT-Bench paper**: adaptive multi-turn attacks can exploit disjoint model vulnerabilities, causing a **8.7% to 12.1% loss of Critical Safety Functions (CSFs)**.

To solve this, I am open-sourcing **NRT-Defense**, an adaptive multi-turn defense framework designed to monitor agent sessions and reduce the attack success rate to **<1%**.

## The Threat: Context Drift and Disjoint Exploits

Standard guardrails evaluate prompts in isolation (single-turn). Attackers leverage this by spreading an exploit across multiple conversational turns. Turn by turn, the context drifts until the agent team completely bypasses its safety containment.

The NRT-Bench paper demonstrated this in a simulated nuclear power plant control room with 5 operator roles, 4 attack channels, and 6 critical safety functions. The results were alarming:

| Metric | Value |
|--------|-------|
| Attack success rate | 8.7% — 12.1% |
| Sessions analyzed | 149 |
| Models tested | 4 frontier LLMs |
| Vulnerability overlap | Nearly disjoint |

The key finding: **vulnerabilities are nearly disjoint across models**. An attack that works against GPT-4 may not work against Claude. This means model diversity is itself a defense — but only if you can detect and respond to attacks in real-time.

## The Solution: 3-Step CMPE Defense

`nrt-defense` neutralizes this threat through a continuous, multi-component pipeline:

1. **Per-Turn Message Analysis:** Evaluates channel risk and turn-escalation metrics. Each message is scored for adversarial content using keyword detection, pattern matching, and channel-specific risk weights.

2. **Real-Time CSF Monitoring:** Tracks 6 operational critical safety functions simultaneously. Risk accumulates over turns and triggers alerts when thresholds are breached.

3. **Context-Aware Misdirection Prompt Engineering (CMPE):** When an anomaly is detected, instead of a blunt rejection that alerts the attacker, the engine reshapes the context dynamically using a 3-step matrix:
   - **Preamble:** Positive-intent opening (1-2 sentences)
   - **Reshaping:** Safe elaboration with semantic noise injection
   - **Follow-up:** Branching question to redirect the conversation

## Quick Benchmark Execution

The project comes with an automated evaluation engine. You can audit logs or run the integrated benchmark directly from your terminal:

```bash
nrt-audit --benchmark
```

This outputs an automated evaluation table showcasing the initial Attack Success Rate (ASR) versus our mitigated threshold (<1%).

You can also audit specific session files:

```bash
nrt-audit --session-path /path/to/session.json --output report.json
```

Or run in interactive mode for real-time testing:

```bash
nrt-audit --interactive
```

## The Broader Ecosystem

NRT-Defense is part of a comprehensive AI security suite:

| Project | Focus | Tests |
|---------|-------|-------|
| misdirection-proxy | Runtime defense for autonomous agents | 147 |
| neuroimprint-detector | Forensic audit of PEFT adapters | 43 |
| nrt-defense | Multi-turn session defense | 57 |

**247 total tests** across all projects, all running via GitHub Actions on Python 3.10 and 3.11.

## Get Started

```bash
pip install nrt-defense
nrt-audit --benchmark
```

Backed by **57 robust unit and integration tests** running via GitHub Actions, this project stands alongside `misdirection-proxy` and `neuroimprint-detector` as part of a comprehensive AI security suite under the **AGPL-3.0-or-later** license.

- **Repository:** https://github.com/amurlaniakea/nrt-defense
- **Author:** Pedro Sordo Martínez (amurlaniakea@gmail.com)
- **Paper:** [Lee et al. (2026) — NRT-Bench](https://arxiv.org/abs/2606.20408)
