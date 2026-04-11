# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please email **ack@abbacus.ai** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Time

- **Acknowledgment:** within 48 hours
- **Assessment:** within 1 week
- **Fix or mitigation:** as soon as practical, prioritized by severity

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Scope

Cortex stores all data locally. Security concerns most relevant to this project include:

- Authentication bypass on the dashboard or API
- Path traversal in MCP tool handlers
- SQL or SPARQL injection
- Command injection via subprocess calls
- Exposure of API keys or credentials

## Disclosure

We will coordinate disclosure with the reporter. Critical fixes will be released as patch versions with a security advisory.
