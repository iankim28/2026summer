# Cursor AI Models Comparison

Practical ranking of models in the Cursor picker (as of July 2026). Sources: [Cursor models & pricing](https://cursor.com/docs/models-and-pricing), [CursorBench](https://cursor.com/cursorbench), per-model docs under [cursor.com/docs/models](https://cursor.com/docs/models/).

## Quick picks

| Goal | Best choice |
|------|-------------|
| **Best bang for buck** | **Cursor Grok 4.5** |
| **Cheapest / fastest daily coding** | **Composer 2.5** |
| **Peak raw quality** | **Fable 5** |
| **Strong mid-tier (API pool)** | **Sonnet 5** or **GPT-5.6 Terra** |
| **Hard multi-step (API pool)** | **Opus 4.8** or **GPT-5.6 Sol** |

## Effectiveness (capability)

Rough order from Cursor’s positioning / CursorBench:

1. **Fable 5** — top of CursorBench; best for hard, long agent runs
2. **GPT-5.6 Sol** / **Opus 4.8** / **Grok 4.5** — frontier tier; close enough that cost and feel matter more than the score gap
3. **Sonnet 5** / **GPT-5.6 Terra** — near-frontier for most coding work
4. **Composer 2.5** — very good for IDE agent work, not peak reasoning
5. **Sonnet 4.6** / **Opus 4.6** — older; skip unless you need them

## Cost (cheap → expensive)

1. **Composer 2.5** — cheapest; first-party pool
2. **Grok 4.5** — still first-party; much cheaper per task than Opus/Sol on CursorBench
3. **Sonnet 5** (promo) / **Terra**
4. **Opus 4.8** / **Opus 4.6**
5. **GPT-5.6 Sol**
6. **Fable 5** — ~2× Opus

Grok and Composer draw from Cursor’s **first-party** usage pool (usually more generous than third-party API models).

### Approximate API rates ($ / 1M tokens)

| Model | Input | Output | Notes |
|--------|------:|-------:|--------|
| Composer 2.5 | $0.50 | $2.50 | Fast variant: $3 / $15 (default in product) |
| Grok 4.5 | $2 | $6 | Fast: $4 / $18 |
| Sonnet 5 | $3 → promo **$2** | $15 → promo **$10** | Promo through Aug 31, 2026 |
| Sonnet 4.6 | $3 | $15 | |
| GPT-5.6 Terra | $2.50 | $15 | Fast = 2× |
| Opus 4.8 / 4.6 | $5 | $25 | Opus Fast available (higher rates) |
| GPT-5.6 Sol | $5 | $30 | Fast = 2× |
| Fable 5 | $10 | $50 | ~2× Opus |

## Speed / latency

- **Fast** in the picker = priority serving (usually **higher $/token**), not “smarter.”
- **Composer 2.5 Fast** — best for snappy interactive edits
- **Grok 4.5 High Fast** — capable + responsive; good default for agent sessions
- **High / Medium** — effort/thinking level, not a global intelligence ranking across vendors

## What to use when

| Situation | Model |
|-----------|--------|
| Daily coding / notebooks / refactors | Composer 2.5, or Grok 4.5 for more capability at low extra cost |
| Hard bugs, multi-file design, long agents | Grok 4.5 → escalate to Opus 4.8 or Sol if it stalls |
| Max quality, cost secondary | Fable 5 |
| Avoid for new work | Sonnet 4.6 / Opus 4.6 (superseded by Sonnet 5 / Opus 4.8) |

## Bottom line

For effectiveness-per-dollar, **Grok 4.5** is the best default. Use **Composer** when you want cheaper/faster iteration, and **Fable / Opus / Sol** only when you need the last bit of quality.

## Models in the picker

| Model | Tags | Role |
|--------|------|------|
| Cursor Grok 4.5 | High Fast (NEW) | First-party flagship; hard long-running tasks |
| Composer 2.5 | Fast | First-party everyday agentic coding |
| Opus 4.8 | High | Anthropic strongest for autonomous multi-step work |
| GPT-5.6 Sol | Medium | Strongest GPT-5.6; long agent sessions |
| Fable 5 | High | Highest CursorBench; ~2× Opus cost |
| Sonnet 5 | High | Near-Opus quality, much cheaper |
| GPT-5.6 Terra | Medium | Mid GPT-5.6; ~half Sol’s price |
| Sonnet 4.6 | Medium | Older Sonnet; prefer Sonnet 5 |
| Opus 4.6 | High | Previous flagship; prefer Opus 4.8 |
