# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.0   | Yes       |

## Reporting a Security Vulnerability

If you discover a security vulnerability in NRT-Defense, please report it responsibly.

**Do NOT open a public GitHub Issue for security vulnerabilities.**

Instead, report via email:
- **Email:** amurlaniakea@gmail.com
- **Subject:** `[SECURITY] NRT-Defense vulnerability`

You will receive a response within 48 hours.

## Security Considerations

NRT-Defense is a defense framework for LLM agents in safety-critical systems:

- **Per-Turn Analyzer** uses keyword and pattern matching. Novel attack techniques may evade detection.
- **Adaptive Misdirection** responses are heuristic. Sophisticated attackers may distinguish them from genuine responses.
- **CSF State Monitoring** tracks 6 Critical Safety Functions but cannot prevent all safety violations.
- **Multi-Model Vulnerability Mapping** relies on known vulnerability profiles. Zero-day vulnerabilities are not covered.

**Use NRT-Defense as one layer in a defense-in-depth strategy.**

## Dependencies

Runtime: `numpy`
Dev: `pytest`, `pytest-cov`
