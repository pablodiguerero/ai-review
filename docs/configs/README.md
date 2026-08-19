# 📘 AI Review Configuration

AI Review supports multiple configuration formats and sources. All of them are automatically detected at runtime.

---

## 📂 Supported formats

- **YAML** (recommended): `.ai-review.yaml`
- **JSON**: `.ai-review.json`
- **ENV**: `.env`

👉 You can combine formats: values are loaded in order of priority.

---

## 📑 Load priority

1. **YAML** (`.ai-review.yaml` or path from `AI_REVIEW_CONFIG_FILE_YAML`)
2. **JSON** (`.ai-review.json` or path from `AI_REVIEW_CONFIG_FILE_JSON`)
3. **ENV** (`.env` or path from `AI_REVIEW_CONFIG_FILE_ENV`)
4. **Environment variables** (`LLM__PROVIDER=OPENAI`, etc.)
5. **Initialization arguments** (if used as a library)

---

## ⚙️ Override file paths

You can override default config locations using environment variables:

- `AI_REVIEW_CONFIG_FILE_YAML` — path to `.yaml` config
- `AI_REVIEW_CONFIG_FILE_JSON` — path to `.json` config
- `AI_REVIEW_CONFIG_FILE_ENV` — path to `.env`

By default, configs are loaded from the **project root**.

---

## 🤖 OpenAI-compatible API selection

`LLM__META__API` picks which OpenAI-compatible endpoint the client talks to:

- `AUTO` (default) — keeps the existing prefix rule: `gpt-5*`/`gpt-4.1*` use `/responses`, everything else uses
  `/chat/completions`
- `CHAT` — always use `/chat/completions`, regardless of model name
- `RESPONSES` — always use `/responses`, regardless of model name

Streaming (`LLM__META__STREAM=true`) is only supported on `/chat/completions`; it is rejected at startup when
`use_responses_api` resolves to `true` (i.e. `AUTO` with a `gpt-5`/`gpt-4.1` model, or an explicit `RESPONSES`).

This matters for gateways that route different models to different endpoints, e.g. the OpenCode Go gateway
(`LLM__HTTP_CLIENT__API_URL=https://opencode.ai/zen/go/v1`), where `deepseek-v4-flash` is only served on
`/chat/completions` (`LLM__META__API=CHAT`) and `grok-4.5` is only served on `/responses`
(`LLM__META__API=RESPONSES`). See [../ci/gitlab-opencode-go.md](../ci/gitlab-opencode-go.md).

`LLM__HTTP_CLIENT__CONNECT_TIMEOUT` sets the TCP connect timeout (seconds, default `10`) separately from
`LLM__HTTP_CLIENT__TIMEOUT`, which covers read/write/pool.

`LLM__META__EXTRA_BODY` is a JSON object merged into the outgoing OpenAI-compatible request body (top-level keys),
letting you pass vendor-specific fields the client doesn't model directly. Keys are rejected at config load time if
they name one of the fields AI Review manages itself — `stream`, `stream_options`, `messages`, `input`, `model`,
`response_format`, `text` — since overriding those would bypass the client's own request construction (including the
streaming-support check). Any other key, e.g. a vendor's reasoning/thinking controls, is merged in as-is. Examples
for the OpenCode Go gateway:

- DeepSeek on `/chat/completions` (`LLM__META__API=CHAT`) — disable reasoning to cut latency and token spend:
  `LLM__META__EXTRA_BODY={"thinking":{"type":"disabled"}}`
- Grok on `/responses` (`LLM__META__API=RESPONSES`) — lower reasoning effort:
  `LLM__META__EXTRA_BODY={"reasoning":{"effort":"low"}}`

---

## 🛡️ Agent and review hardening settings

- `AGENT__DEADLINE_SECONDS` (int, default `None`) — a soft wall-clock budget for the agent loop, in seconds. Once
  elapsed, the loop stops calling tools and moves straight to a forced final answer. It is a soft bound: one
  in-flight tool/LLM request plus the force-final attempts themselves can still run past it.
- `AGENT__FALLBACK_TO_DIRECT_CHAT` (bool, default `false`) — this is a behaviour change from upstream: when the
  agent loop fails, nothing is posted unless this is set to `true`, in which case AI Review falls back to a direct
  (non-agent) chat call for the review.
- `REVIEW__FAIL_ON_EMPTY_RESULT` (bool, default `false`) — when `true`, the `run`, `run-inline`, and `run-summary`
  commands exit with code `1` if the LLM produced no usable review. A run that is skipped because there was nothing
  to review still exits `0`.
- `AGENT__MAX_COMMAND_OUTPUT_CHARS` default changed from `40000` to `8000`.
- `AGENT__ALLOW_COMMANDS` default gained read-only `head`, `tail`, `wc`, `sed -n 'A,Bp' FILE`, and `find` (without
  its mutating flags) patterns. Regardless of the allowlist, unquoted shell operators (`|`, `&&`, `;`, `>`, etc.)
  and newlines inside a command are always rejected.

`LLM__HTTP_CLIENT__CONNECT_TIMEOUT` is honoured by the OpenAI-compatible clients only; other providers ignore it.

The sample files [.env.example](./.env.example), [.ai-review.yaml](./.ai-review.yaml), and [../ci](../ci) `*.yaml`
templates still show the previous defaults — they were intentionally left untouched.

🔒 **Security note:** `.ai-review.yaml` / `.ai-review.json` / `.env` are read from the working directory by
default (see "Load priority" above) and take priority over environment variables, so when AI Review runs against a
checkout it doesn't otherwise control — e.g. reviewing a merge request's own working copy in CI — that checkout can
add or edit one of those files to reconfigure the reviewer itself (widen the agent's command allowlist, redirect
`LLM_`/`VCS_` API URLs to exfiltrate tokens, etc.). Point `AI_REVIEW_CONFIG_FILE_YAML`, `AI_REVIEW_CONFIG_FILE_JSON`,
and `AI_REVIEW_CONFIG_FILE_ENV` at a path that doesn't exist to neutralise this — a missing config file is skipped,
so only environment variables (or CLI init args) configure the run. See
[../ci/gitlab-opencode-go.md](../ci/gitlab-opencode-go.md) for a worked example.

---

## 📘 Examples

- [.ai-review.yaml](./.ai-review.yaml) — main YAML config with comments
- [.ai-review.json](./.ai-review.json) — JSON config example
- [.env.example](./.env.example) — ENV config example

---

## 🔍 Tips

- Use **YAML** for most projects — it’s human-friendly and supports comments.
- **JSON** is convenient for automation (e.g., CI/CD pipelines).
- **ENV** is useful for local development and quick overrides.
