# Security Policy

## Reporting a Vulnerability

Please do not disclose credentials, personal resumes, or exploit details in a
public issue. Report security concerns privately to `blues924@outlook.com` with:

- the affected file or component;
- steps to reproduce the issue;
- the potential impact;
- a suggested fix, if available.

Do not include a live API key in the report. A redacted prefix and provider name
are sufficient for identifying a credential.

## Credential Handling

- Copy `.env.example` to `.env` for local configuration.
- Keep `API_KEY` empty in committed examples.
- Never put credentials in source code, logs, screenshots, or test fixtures.
- Rotate a credential immediately if it has been committed, even when the file
  is later deleted. Deleting a file does not remove it from Git history.

## Personal Data

Real resumes, generated `resume.json` files, and conversation logs must remain
local. Before publishing recommendation events or datasets, scan them for names,
email addresses, phone numbers, identifiers, and other personal information.
