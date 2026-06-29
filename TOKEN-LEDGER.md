# TOKEN LEDGER — permission tokens (James's rule, started 2026-06-29)

Start balance: **10**. A token is spent only on permission-requiring actions
(external sends, irreversible ops, config changes, personal-data reads). Reversible
internal work is free. Decrement and log every spend here.

| # | time (EDT) | action | tokens | balance after |
|---|-----------|--------|--------|---------------|
| — | 08:1x | (ledger opened; day plan set) | 0 | 10 |
| 1 | 11:0x | Block B: emailed "Petrichor — progress 1" to jamestrichmond@gmail.com (himalaya, exit 0) | 1 | 9 |

## Spend log (append-only, one line per spend)
- 08:1x EDT — Ledger opened. Balance 10/10. Day plan: progress emails (B,D), private repo push (C), homecoming brief email (E); reserve the rest.
- 11:0x EDT — SPEND 1 (external send). Block B progress email sent via himalaya, exit 0. Demo run clean + RESULTS.md + figures regenerated (all free/internal). Balance 9/10.
