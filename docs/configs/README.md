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

Streaming (`LLM__META__STREAM=true`) is supported on both `/chat/completions` and `/responses`. On `/responses`,
the client parses the SSE event stream and accumulates `output_text` deltas; a stall or dropped connection before
any content has arrived is retried once at zero cost — nothing billable was generated yet, mirroring how the
`/chat/completions` client already handles a mid-stream drop. This is useful for gateways/models where the first
`/responses` request occasionally stalls for the full read timeout with no bytes at all (a repeat request usually
answers in seconds): pair `LLM__META__STREAM=true` with a shorter `LLM__HTTP_CLIENT__TIMEOUT` so a stall is detected
and retried well before the job's own timeout. See [../ci/gitlab-opencode-go.md](../ci/gitlab-opencode-go.md) for a
worked example (grok on the OpenCode Go gateway).

This matters for gateways that route different models to different endpoints, e.g. the OpenCode Go gateway
(`LLM__HTTP_CLIENT__API_URL=https://opencode.ai/zen/go/v1`), where `deepseek-v4-flash` is only served on
`/chat/completions` (`LLM__META__API=CHAT`) and `grok-4.5` is only served on `/responses`
(`LLM__META__API=RESPONSES`). See [../ci/gitlab-opencode-go.md](../ci/gitlab-opencode-go.md).

`LLM__HTTP_CLIENT__CONNECT_TIMEOUT` sets the TCP connect timeout (seconds, default `10`) separately from
`LLM__HTTP_CLIENT__TIMEOUT`, which covers read/write/pool.

`LLM__META__EXTRA_BODY` is a JSON object merged into the outgoing OpenAI-compatible request body (top-level keys),
letting you pass vendor-specific fields the client doesn't model directly. Keys are rejected at config load time if
they name one of the fields AI Review manages itself — `stream`, `stream_options`, `messages`, `input`, `model`,
`response_format`, `text` — since overriding those would bypass the client's own request construction. Any other
key, e.g. a vendor's reasoning/thinking controls, is merged in as-is. Examples for the OpenCode Go gateway:

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
  its mutating flags) patterns. Regardless of the allowlist, unquoted shell operators (`|`, `&&`, `;`, `>`, etc.),
  newlines, and `/proc/` or `/dev/` paths inside a command are always rejected.
- `REVIEW__SUMMARY_HEADER` (string, default `""`) and `REVIEW__SUMMARY_REPLACE_PREVIOUS` (bool, default `false`)
  together keep one summary comment per model per MR across reruns instead of piling up duplicates.
  `REVIEW__SUMMARY_HEADER` is a `str.format`-style template rendered with `model=<LLM__META__MODEL>` and prepended
  to the summary body when non-empty, e.g. `REVIEW__SUMMARY_HEADER="### AI review: {model}"` labels the comment
  with the model that wrote it — useful when several jobs (different models) each post their own summary to the
  same MR. `REVIEW__SUMMARY_REPLACE_PREVIOUS=true` makes the summary runner update its own most recent prior
  comment (matched by the `REVIEW__SUMMARY_TAG`) in place instead of creating a new one, and delete any older
  duplicates from earlier runs; if the VCS can't update a comment in place, it falls back to creating a new one.
  On GitLab this edits the note directly; on GitHub, Gitea, Bitbucket Cloud/Server, and Azure DevOps — which have
  no note-update API — it deletes the old comment and creates a new one. A comment is matched to its tag (e.g.
  `REVIEW__SUMMARY_TAG`) only when some line of the body, once stripped, equals the tag exactly, not by a
  substring search, so a comment that merely mentions another tag in its prose is never picked up.
- `REVIEW__SUMMARY_NORMALIZE_TABLES` (bool, default `true`) — before posting, the summary text is passed through a
  deterministic Markdown-table fixer that turns pipe-separated lines lacking outer pipes and/or a `| --- | --- |`
  separator row into valid GFM tables (padding short rows, trimming long ones, and inserting exactly one blank line
  before/after the table), so tables like a "Clean Code Evaluation Table" render correctly on GitLab/GitHub instead
  of as raw pipe-separated text. Content inside fenced ` ``` ` code blocks is never touched. Set to `false` to post
  the model's raw Markdown unchanged.
- `AGENT__CHECKPOINT_DIR` (path, default `None`) — when set, the agent loop persists its progress to
  `<dir>/<sha256 of a key>.json` after every iteration. The checkpoint key is
  `<project_id>:<merge_request_id>:<LLM__META__MODEL>:<REVIEW__SUMMARY_TAG>` — it deliberately does **not** include
  `head_sha`, so a single checkpoint file forms a **chain** that evolves across the whole life of an MR (every job
  retry and every new push to the same MR resumes the same chain instead of starting from scratch). The current
  `head_sha` is stored inside the checkpoint payload and compared against the commit being reviewed on each run.
  A checkpoint carries a `stage` that drives what the next run does:
  - `investigating` — the loop was mid-run (crash/timeout) on the current commit; the next run continues from the
    next iteration with the same traces, signatures and counters (cheapest replay, no new work required).
  - `needs_final` — the loop exhausted its iterations on the current commit without a `FINAL`; if the commit hasn't
    changed, the next run skips straight to a single fresh force-final pass and republishes (the summary runner's
    `REVIEW__SUMMARY_REPLACE_PREVIOUS` then updates the existing comment instead of duplicating it).
  - `reviewed` — the loop already produced and posted a review. If the commit hasn't changed, a re-run restarts a
    new investigation round while keeping the prior tool traces and signatures (so it won't re-run identical
    commands, but it must still execute at least `AGENT__RESUME_MIN_NEW_TOOL_CALLS` new read-only commands before a
    `FINAL` is accepted again). If the commit *has* changed (a new push), the prior round's tool traces are folded
    into a compact text synopsis (kept under the `## Earlier findings (previous rounds)` heading in the prompt),
    traces and signatures are reset, and a full fresh investigation round runs against the new diff — the model
    still remembers what earlier rounds found without re-paying for every tool call. Either way `round` increments
    by one.
  - `needs_final`/`investigating` on a *different* `head_sha` are treated the same as `reviewed` + new commit: the
    evidence is folded into the synopsis and a fresh round starts, because the code underneath it changed.
  `AGENT__RESUME_MIN_NEW_TOOL_CALLS` (int, default `2`) — when a run starts a new investigation round on top of a
  resumed checkpoint, the model must execute at least this many *new* tool commands (in addition to any inherited
  `AGENT__MIN_TOOL_CALLS` floor) before a `FINAL` is accepted; a premature `FINAL` is nudged back into the loop the
  same way an `AGENT__MIN_TOOL_CALLS` violation is. `AGENT__MAX_TRACE_HISTORY` (int, default `16`) — the maximum
  number of raw tool traces kept in a checkpoint/prompt at once; older traces are folded into the running synopsis
  instead of being dropped, so long chains stay bounded in size without losing earlier findings. The checkpoint is
  never deleted on its own — it is overwritten in place — so a directory outside the job's own throwaway workspace
  (e.g. a runner-persistent cache mount) is required for it to survive a retry; a broken/unreadable checkpoint file
  is logged and ignored, and a fresh save simply overwrites it. Only the summary review uses checkpointing today.
  See [../ci/gitlab-opencode-go.md](../ci/gitlab-opencode-go.md) for a mounted cache example.

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
