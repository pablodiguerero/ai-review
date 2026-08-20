# AI Review

AI-powered code review CLI. It runs in CI, reads the diff of a merge/pull request, drives an agent that verifies its
own claims against the real repository with read-only shell commands, and posts the result as a review comment.

This is a fork of [Nikita-Filonov/ai-review](https://github.com/Nikita-Filonov/ai-review) that has diverged into a
security- and agent-hardened build focused on running against the OpenCode Go / Zen gateway in CI.

Reviews are advisory: the tool posts comments for a human to read, it does not approve, block, or merge anything by
itself. Whether a pipeline treats a failed review job as blocking is a CI configuration choice (`allow_failure`), not
something this tool decides.

## Table of Contents

- [How it works](#how-it-works)
- [LLM providers](#llm-providers)
- [VCS providers](#vcs-providers)
- [CLI commands](#cli-commands)
- [Agent hardening and reliability](#agent-hardening-and-reliability)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [License](#license)

## How it works

1. AI Review reads the merge/pull request diff from the configured VCS provider, using the review mode set by
   `REVIEW__MODE` (e.g. `FULL_FILE_DIFF`, `ONLY_ADDED_WITH_CONTEXT`, …).
2. When agent mode is enabled (`AGENT__ENABLED=true`), the model runs a ReAct-style loop: it can issue read-only
   shell commands (`ls`, `cat`, `head`, `tail`, `wc`, `sed -n`, `rg`, `grep`, `find`, and a handful of read-only
   `git` subcommands) against the checked-out repository to verify its own claims — checking that a referenced
   function actually exists, that an import is really unused, that a "missing" test file is really missing — before
   writing anything down. This is deeper than a single-shot call: the model is required to gather evidence before
   it is allowed to finish.
3. The agent's `FINAL` answer becomes the review body. Only a well-formed `FINAL` step is ever posted; a step that
   doesn't parse or isn't `FINAL` is dropped rather than posted as-is.
4. The result is posted back to the merge/pull request as a summary comment, and/or as inline comments, a
   cross-file context review, or a reply into an existing discussion thread, depending on which command is run.

## LLM providers

Configured via `LLM__PROVIDER` (see `ai_review/services/llm/`):

- `OPENAI` — any OpenAI-compatible `/chat/completions` or `/responses` endpoint, including gateways. This is the
  provider used for the OpenCode Go / Zen gateway.
- `CLAUDE`
- `GEMINI`
- `OLLAMA` — local/self-hosted, for fully offline reviews
- `BEDROCK`
- `OPENROUTER`
- `AZURE_OPENAI`

For the `OPENAI` provider, three `LLM__META__*` settings matter for gateway routing:

- `LLM__META__API` — `AUTO` (default, picks `/responses` for `gpt-5*`/`gpt-4.1*` model prefixes and
  `/chat/completions` for everything else) | `CHAT` (force `/chat/completions`) | `RESPONSES` (force `/responses`).
  Needed when a gateway serves different models on different endpoints and the prefix heuristic doesn't apply.
- `LLM__META__STREAM` (bool, default `false`) — supported on both `/chat/completions` and `/responses`. On
  `/responses` the client parses the SSE stream and accumulates `output_text` deltas; a stall before any content
  arrives is retried once at zero cost. Useful for gateways where the first request to a given model occasionally
  stalls for the full read timeout.
- `LLM__META__EXTRA_BODY` — a JSON object merged into the outgoing request body, for vendor-specific fields (e.g. a
  model's reasoning/thinking controls). Keys that would override a field AI Review manages itself (`stream`,
  `stream_options`, `messages`, `input`, `model`, `response_format`, `text`) are rejected at config load time.

See [docs/ci/gitlab-opencode-go.md](./docs/ci/gitlab-opencode-go.md) for the full routing setup against the
OpenCode Go gateway (`https://opencode.ai/zen/go/v1`), including which models need `CHAT` vs `RESPONSES` and the
time-budget arithmetic for the agent deadline.

## VCS providers

Configured via `VCS__PROVIDER` (see `ai_review/services/vcs/`):

- `GITLAB`
- `GITHUB`
- `BITBUCKET_CLOUD`
- `BITBUCKET_SERVER`
- `GITEA`
- `AZURE_DEVOPS`

## CLI commands

All commands are defined in `ai_review/cli/main.py` and exposed through the `ai-review` entry point:

| Command | Description |
|---|---|
| `ai-review run` | Runs the full pipeline: inline review followed by summary review. |
| `ai-review run-inline` | Posts line-by-line inline comments on the diff. |
| `ai-review run-context` | Runs a broader, cross-file review without posting per-line comments. |
| `ai-review run-summary` | Posts a single summary comment for the whole change. |
| `ai-review run-inline-reply` | Generates AI replies to existing inline comment threads. |
| `ai-review run-summary-reply` | Generates an AI reply to the existing summary review thread. |
| `ai-review clear-inline` | Deletes all AI Review inline comments from the current merge/pull request. |
| `ai-review clear-summary` | Deletes all AI Review summary comments from the current merge/pull request. |
| `ai-review show-config` | Prints the fully resolved configuration as JSON — use this to check what a run will actually do before it runs. |

## Agent hardening and reliability

The parts of this fork that differ meaningfully from upstream:

- **Read-only command allowlist.** `AGENT__ALLOW_COMMANDS` only permits `ls`, `cat`, `head`, `tail`, `wc`,
  `sed -n 'A,Bp' FILE`, `rg` (excluding `--pre`/`--pre-glob`/`--hostname-bin`/`--search-zip`/`-z`), `grep`, `find`
  (excluding its mutating flags: `-exec`, `-execdir`, `-ok`, `-okdir`, `-delete`, `-fprint`, `-fprintf`, `-fls`),
  and read-only `git` subcommands (`status`, `show`, `diff`, `log`, `rev-parse`, `ls-files`, excluding `--output`).
  Regardless of the allowlist, unquoted shell operators (`|`, `&&`, `;`, `>`, etc.), embedded newlines, and
  `/proc/`/`/dev/` paths inside a command are always rejected.
- **`AGENT__MIN_TOOL_CALLS`** (default `0`) — a floor on how many tool calls the agent must execute before a
  `FINAL` answer is accepted, so it can't skip verification and go straight to an opinion.
- **`AGENT__DEADLINE_SECONDS`** (default unset) — a soft wall-clock budget for the whole agent loop; once it
  elapses, the loop stops calling tools and moves to a forced final answer.
- **`AGENT__FORCE_FINAL_ATTEMPTS`** (default `2`) — how many forced-final attempts are made if the model won't
  produce a well-formed `FINAL` on its own (e.g. after the deadline, or after exhausting `AGENT__MAX_ITERATIONS`).
- **`AGENT__MAX_COMMAND_OUTPUT_CHARS`** (default `8000`, tightened from upstream's `40000`) — caps how much output
  from a single tool call is fed back into the model's context.
- **Checkpoint chain** (summary review only) — `AGENT__CHECKPOINT_DIR` persists the agent's progress after every
  iteration, keyed by `<project_id>:<merge_request_id>:<model>:<summary_tag>` (deliberately not including the
  commit SHA). This lets a retried job resume an in-progress loop instead of restarting it, and lets a new push to
  the same MR advance into a fresh review round while reusing the prior round's findings — folded into a compact
  synopsis — instead of re-running every command from scratch. `AGENT__RESUME_MIN_NEW_TOOL_CALLS` (default `2`)
  requires that many *new* tool calls before a resumed round accepts `FINAL`; `AGENT__MAX_TRACE_HISTORY` (default
  `16`) bounds how many raw tool traces are kept before older ones are folded into the synopsis.
- **One summary comment per model.** `REVIEW__SUMMARY_HEADER` (a `str.format` template rendered with `model=...`,
  e.g. `"### AI review: {model}"`) labels which model wrote a summary comment, and
  `REVIEW__SUMMARY_REPLACE_PREVIOUS` makes the summary runner update its own most recent comment in place — matched
  by `REVIEW__SUMMARY_TAG` as a standalone line, never a substring — instead of piling up duplicates across reruns.
- **Markdown table normalization.** `REVIEW__SUMMARY_NORMALIZE_TABLES` (default `true`) rewrites pipe-separated
  text lacking outer pipes or a separator row into valid GFM tables before posting, so tables the model writes
  loosely still render correctly on GitLab/GitHub. Fenced code blocks are left untouched.
- **`REVIEW__FAIL_ON_EMPTY_RESULT`** (default `false`) — when `true`, `run`/`run-inline`/`run-summary` exit `1` if
  the LLM produced no usable review, instead of always exiting `0`.
- **Config-file pins.** `.ai-review.yaml`/`.ai-review.json`/`.env` are read from the working directory by default
  and take priority over environment variables — which means a reviewed checkout could otherwise add or edit one of
  those files to reconfigure the reviewer itself (widen the command allowlist, redirect the LLM/VCS API URL to
  exfiltrate tokens). Pointing `AI_REVIEW_CONFIG_FILE_YAML`, `AI_REVIEW_CONFIG_FILE_JSON`, and
  `AI_REVIEW_CONFIG_FILE_ENV` at a path that doesn't exist neutralizes this: a missing config file is skipped, so
  only environment variables configure the run. See [docs/ci/gitlab-opencode-go.md](./docs/ci/gitlab-opencode-go.md)
  for a worked example.
- **Local e2e harness.** `e2e/run_docker_review.sh` runs the actual Docker image (never the host Python) against a
  real GitLab merge request in dry-run mode and checks the result with `e2e/assert_review.py`, for testing prompt,
  model, or agent-setting changes before they ship. See [docs/e2e.md](./docs/e2e.md).

## Quick start

AI Review ships as a Docker image; there is no local install step. Build/pull the image, then run it with your LLM
and VCS credentials as environment variables. Minimal example, reviewing a GitLab merge request:

```bash
export LLM__HTTP_CLIENT__API_TOKEN=sk-...
export VCS__HTTP_CLIENT__API_TOKEN=glpat-...

docker run --rm \
  -v "$(pwd)":/app -w /app \
  -e LLM__HTTP_CLIENT__API_TOKEN \
  -e VCS__HTTP_CLIENT__API_TOKEN \
  -e LLM__PROVIDER=OPENAI \
  -e LLM__META__MODEL=gpt-4o-mini \
  -e LLM__HTTP_CLIENT__API_URL=https://api.openai.com/v1 \
  -e VCS__PROVIDER=GITLAB \
  -e VCS__PIPELINE__PROJECT_ID="$CI_PROJECT_ID" \
  -e VCS__PIPELINE__MERGE_REQUEST_ID="$CI_MERGE_REQUEST_IID" \
  -e VCS__HTTP_CLIENT__API_URL="$CI_SERVER_URL" \
  ghcr.io/pablodiguerero/ai-review:<sha> \
  ai-review run-summary
```

Passing `-e VAR` without a value (rather than `-e VAR=$VAR`) keeps the secret out of the process list and shell
history of anything inspecting the `docker run` invocation itself.

For a complete CI job — agent mode, the OpenCode Go gateway, checkpointing, and the `AI_REVIEW_CONFIG_FILE_*`
security pins — see [docs/ci/gitlab-opencode-go.md](./docs/ci/gitlab-opencode-go.md).

Configuration can also come from a `.ai-review.yaml` or `.ai-review.json` file instead of (or combined with)
environment variables — see [Configuration](#configuration). Run `ai-review show-config` to print the fully
resolved configuration and confirm what a run will actually do.

## Configuration

Every setting shown above, plus timeouts, prompts, artifacts, and logging, is documented in full in
[docs/configs/README.md](./docs/configs/README.md), including load order between YAML/JSON/ENV/environment
variables and the `AI_REVIEW_CONFIG_FILE_*` override paths.

## Documentation

- [docs/ci](./docs/ci) — CI/CD integration templates (GitHub Actions, GitLab CI, Bitbucket Pipelines, Jenkins,
  Azure Pipelines), including [docs/ci/gitlab-opencode-go.md](./docs/ci/gitlab-opencode-go.md) for the OpenCode Go
  gateway setup
- [docs/cli](./docs/cli) — CLI command reference and usage examples
- [docs/configs](./docs/configs) — full configuration reference and example `.yaml`/`.json`/`.env` files
- [docs/hooks](./docs/hooks) — lifecycle hooks reference
- [docs/prompts](./docs/prompts) — prompt templates (Python/Go, light/strict) and prompt variable reference
- [docs/troubleshooting](./docs/troubleshooting) — common environment and Git-related issues
- [docs/e2e.md](./docs/e2e.md) — local Docker-based end-to-end harness for testing changes against a real MR

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
