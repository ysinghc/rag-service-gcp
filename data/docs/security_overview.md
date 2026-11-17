# Security overview

All customer data is encrypted in transit with TLS 1.2 or newer and encrypted at rest with AES-256. Encryption keys are rotated every 90 days.

Single sign-on is available through SAML 2.0 and OpenID Connect on the Pro and Enterprise plans. Multi-factor authentication is required for every workspace administrator account.

Audit logs record every login, export and permission change. Audit logs are retained for 400 days and can be streamed to a SIEM over HTTPS webhooks.

Data residency: customers on the Enterprise plan can keep all customer data in the europe-west1 region. Backups stay in the same region.

Vulnerability disclosure: security reports can be submitted through the responsible disclosure program. We acknowledge every report within two business days.
