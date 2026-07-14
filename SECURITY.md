# Security Policy

## Supported versions

Security fixes are applied to the latest release and the default branch. Older releases are not maintained.

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/connorg45/hotbatch/security/advisories/new) instead of opening a public issue. Include the affected version, reproduction steps, expected impact, and any suggested mitigation.

Do not include credentials, private model data, or sensitive prompts in a report. You should receive an acknowledgement within seven days. Timelines for validation and remediation depend on severity and reproducibility.

## Deployment scope

Hotbatch does not provide built-in authentication or per-client rate limiting. Operators exposing it outside a trusted network should use a gateway that supplies authentication, throttling, and TLS, and should choose queue and generation limits appropriate for the host.
