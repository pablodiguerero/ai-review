# End-to-end docker harness

`e2e/run_docker_review.sh` runs the `ai-review` **docker image** (never the
host Python) against a real GitLab merge request in dry-run mode, the same
way consumer CI pipelines do, and checks the result with
`e2e/assert_review.py` by reading the JSON artifacts the run produced. It is
meant to be repeatable locally: rerun it after changing prompts, models, or
agent settings and diff the metrics.

## Prerequisites

- Docker Desktop or [colima](https://github.com/abiosoft/colima) on macOS.
  colima's bind mounts only work for paths under `$HOME` — keep
  `E2E_WORKSPACE` there (the default already is).
- A GitLab **read-only** token is enough: the harness always sets
  `REVIEW__DRY_RUN=true`, so it reads the merge request but never posts a
  comment. Export it as `AI_REVIEW_GITLAB_TOKEN`.
- An `OPENCODE_API_KEY` for the LLM gateway (default gateway is
  `https://opencode.ai/zen/go/v1`).
- `python3.12` on the host for `assert_review.py` (stdlib only). If it is not
  on `PATH`, set `E2E_PYTHON` to a Python >= 3.10 interpreter; the script
  refuses to run `assert_review.py` under anything older because it relies on
  modern union-type syntax.
- `docker`, `git`, `curl` on `PATH`.

## Quick start

```bash
export OPENCODE_API_KEY=...
export AI_REVIEW_GITLAB_TOKEN=...

E2E_BUILD=1 \
E2E_NAME=cabinet \
E2E_PROJECT_ID=4105106 \
E2E_MR_IID=1070 \
E2E_MODEL=gpt-4o-mini \
E2E_REPO_URL=git@gitlab.com:example/cabinet.git \
E2E_PROMPT_FILES='["./.ai-review/prompts/summary.md", "./.ai-review/prompts/verify.md"]' \
E2E_EXPECT_SCORE=1 \
e2e/run_docker_review.sh
```

The script exits with `assert_review.py`'s result (0 pass / 1 fail), not the
docker exit code — the docker exit code is one of the things being asserted.

The container is always run with `AI_REVIEW_CONFIG_FILE_YAML`,
`AI_REVIEW_CONFIG_FILE_JSON` and `AI_REVIEW_CONFIG_FILE_ENV` pinned to
`/nonexistent`, because `ai-review` otherwise loads `.ai-review.yaml` /
`.ai-review.json` / `.env` from the reviewed repo with higher priority than
env vars, which would let a malicious MR reconfigure the reviewer.

## Environment variables

Everything is configured through environment variables; there are no CLI
flags besides `--help`.

### Required

| Variable | Maps to | Notes |
|---|---|---|
| `E2E_NAME` | — | short label, used for the repo clone dir and the run dir prefix |
| `E2E_PROJECT_ID` | `VCS__PIPELINE__PROJECT_ID` | GitLab numeric project id |
| `E2E_MR_IID` | `VCS__PIPELINE__MERGE_REQUEST_ID` | merge request IID (the `!N` number) |
| `E2E_MODEL` | `LLM__META__MODEL` | e.g. `gpt-4o-mini`, `grok-4.5` |
| `OPENCODE_API_KEY` | `LLM__HTTP_CLIENT__API_TOKEN` | secret, passed to docker without ever appearing in `ps`/logs |
| `AI_REVIEW_GITLAB_TOKEN` | `VCS__HTTP_CLIENT__API_TOKEN` | secret, same treatment; also used to fetch MR metadata from the GitLab API |
| `E2E_REPO_LOCAL` and/or `E2E_REPO_URL` | — | required only the first time a given `E2E_NAME` is used (clone source); afterwards the existing clone under the workspace is reused and refreshed |

### Workspace and repo

| Variable | Default |
|---|---|
| `E2E_WORKSPACE` | `$HOME/tmp/ai-review-e2e` (must stay under `$HOME`) |
| `E2E_REPO_LOCAL` | — (local path to clone from; fast, hardlinked) |
| `E2E_REPO_URL` | — (git remote; also used to `git remote set-url origin` on reuse) |
| `E2E_PREPARE` | — (shell command run inside the clone before docker, e.g. `yarn install --frozen-lockfile`) |

### Docker image

| Variable | Default |
|---|---|
| `E2E_IMAGE` | `ai-review:e2e` |
| `E2E_BUILD` | `0` (set to `1` to `docker build` the image from the repo root first) |
| `E2E_EXTRA_DOCKER_ARGS` | — (appended verbatim/word-split to `docker run`, e.g. `-e FOO=bar`) |

### GitLab

| Variable | Default |
|---|---|
| `E2E_GITLAB_API_URL` | `https://gitlab.com` |

### LLM

| Variable | Maps to | Default |
|---|---|---|
| `E2E_API` | `LLM__META__API` | `AUTO` (`AUTO`\|`CHAT`\|`RESPONSES`) |
| `E2E_STREAM` | `LLM__META__STREAM` | `false` |
| `E2E_MAX_TOKENS` | `LLM__META__MAX_TOKENS` | `32000` |
| `E2E_TEMPERATURE` | `LLM__META__TEMPERATURE` | `0.3` |
| `E2E_LLM_API_URL` | `LLM__HTTP_CLIENT__API_URL` | `https://opencode.ai/zen/go/v1` |
| `E2E_LLM_TIMEOUT` | `LLM__HTTP_CLIENT__TIMEOUT` | `240` |
| `E2E_LLM_CONNECT_TIMEOUT` | `LLM__HTTP_CLIENT__CONNECT_TIMEOUT` | `10` |

### Review

| Variable | Maps to | Default |
|---|---|---|
| `E2E_REVIEW_MODE` | `REVIEW__MODE` | `FULL_FILE_DIFF` |
| `E2E_SUMMARY_FEEDBACK_LOOP` | `REVIEW__SUMMARY_FEEDBACK_LOOP` | `true` |
| `E2E_FAIL_ON_EMPTY_RESULT` | `REVIEW__FAIL_ON_EMPTY_RESULT` | unset (only passed through when you set it) |
| `E2E_PROMPT_FILES` | `PROMPT__SUMMARY_PROMPT_FILES` | unset (built-in default prompt is used); pass a JSON list string, e.g. `["./.ai-review/prompts/summary.md", "./.ai-review/prompts/verify.md"]` |

### Agent

| Variable | Maps to | Default |
|---|---|---|
| `E2E_AGENT_ENABLED` | `AGENT__ENABLED` | `true` |
| `E2E_AGENT_MAX_ITERATIONS` | `AGENT__MAX_ITERATIONS` | `30` |
| `E2E_AGENT_MIN_TOOL_CALLS` | `AGENT__MIN_TOOL_CALLS` | `2` |
| `E2E_AGENT_DEADLINE_SECONDS` | `AGENT__DEADLINE_SECONDS` | `300` |
| `E2E_AGENT_FORCE_FINAL_ATTEMPTS` | `AGENT__FORCE_FINAL_ATTEMPTS` | `1` |

### Assertion (forwarded to `assert_review.py`)

| Variable | `assert_review.py` flag | Default |
|---|---|---|
| `E2E_EXPECT_SCORE` | `--expect-score` | off |
| `E2E_MIN_EXECUTED` | `--min-executed N` | unset (check skipped) |
| `E2E_EXPECT_EXIT` | `--expect-exit N` | `0` |
| `E2E_ALLOW_TRACEBACK` | `--allow-traceback` | off |
| `E2E_EXPECT_EMPTY` | `--expect-empty` | off |
| `E2E_ASSERT_JSON` | `--json` | off |

### Misc

| Variable | Default |
|---|---|
| `E2E_PYTHON` | `python3.12` (falls back to `python3` if not found; either must be >= 3.10) |

## Example commands for the three consumer repos

```bash
export OPENCODE_API_KEY=...
export AI_REVIEW_GITLAB_TOKEN=...

E2E_BUILD=1 \
E2E_NAME=cabinet \
E2E_PROJECT_ID=4105106 \
E2E_MR_IID=1070 \
E2E_MODEL=gpt-4o-mini \
E2E_REPO_URL=git@gitlab.com:example/cabinet.git \
E2E_PROMPT_FILES='["./.ai-review/prompts/summary.md", "./.ai-review/prompts/verify.md"]' \
E2E_EXPECT_SCORE=1 \
e2e/run_docker_review.sh

E2E_NAME=expert-app \
E2E_PROJECT_ID=27452488 \
E2E_MR_IID=427 \
E2E_MODEL=gpt-4o-mini \
E2E_REPO_URL=git@gitlab.com:example/expert-app.git \
E2E_PREPARE='yarn install --frozen-lockfile' \
E2E_PROMPT_FILES='["./.ai-review/prompts/summary.md", "./.ai-review/prompts/verify.md"]' \
E2E_EXPECT_SCORE=1 \
e2e/run_docker_review.sh

E2E_NAME=client-app \
E2E_PROJECT_ID=28271363 \
E2E_MR_IID=168 \
E2E_MODEL=gpt-4o-mini \
E2E_REPO_URL=git@gitlab.com:example/client-app.git \
E2E_EXPECT_SCORE=1 \
e2e/run_docker_review.sh
```

`client-app` omits `E2E_PROMPT_FILES` to exercise the built-in default summary
prompt instead of the repos' own prompt overrides.

## Running grok

```bash
E2E_NAME=cabinet \
E2E_PROJECT_ID=4105106 \
E2E_MR_IID=1070 \
E2E_MODEL=grok-4.5 \
E2E_API=RESPONSES \
e2e/run_docker_review.sh
```

## Where output lands

Each run writes to:

```
$E2E_WORKSPACE/runs/<name>-mr<iid>-<model>-<timestamp>/
  run.log            full docker stdout+stderr
  artifacts/llm/*.json
  artifacts/vcs/*.json
  env.txt            non-secret env vars passed to the container
  meta.json          project/MR/model/image/timestamps/docker exit code
```

The repo clone itself lives at `$E2E_WORKSPACE/<name>` and is reused across
runs (fetched and re-checked-out each time) rather than re-cloned.

## Reading the assert output

`assert_review.py --run-dir <dir> ...` prints one `[PASS]`/`[FAIL]` line per
check, a `metrics:` block, and a final `OVERALL: PASS`/`FAIL` line; it exits
1 if any check failed. The checks:

1. docker exit code matches `--expect-exit` (default 0)
2. no `Traceback (most recent call last)` in `run.log`, unless `--allow-traceback`
3. unless `--expect-empty`, at least one LLM artifact must exist and the last
   one (by timestamp) must have a non-empty `data.response`; with
   `--expect-empty`, either no LLM artifacts exist at all (e.g. the agent
   aborted before verifying anything) or the last one has an empty response —
   either way, no `VCS_SUMMARY` artifact may exist, since that would mean
   something was posted
4. the response contains no leaked agent protocol markers (`<TOOL_CALL>`,
   `"action":`, `"TOOL_CALL"`, `</TOOL_CALL>`)
5. with `--expect-score`, the response matches `Overall score:\s*\d`
6. with `--min-executed N`, `data.agent.executed_tool_calls >= N` (read from
   the last LLM artifact, falling back to the `Agent loop summary: ...` log
   line on older images that don't emit the field; if neither is available
   the check fails with "agent stats unavailable")

The `metrics:` block is informational only (not asserted): artifact count,
prompt/response character counts, iteration/tool-call/stop-reason numbers,
wall time, and counts of `Waiting for response` / `blocked by policy` lines
in `run.log`. Pass `--json` (or `E2E_ASSERT_JSON=1`) to get the same
information as a JSON object instead, for building a summary table across
runs.

## Self-testing the checker without a gateway

No API key or GitLab access is needed to validate `assert_review.py` itself:

```bash
python3.12 e2e/assert_review.py --self-test
```

This builds synthetic run directories covering both passing and failing
scenarios (bad exit code, traceback, empty response, leaked protocol
markers, the `Agent loop summary` log fallback, `--expect-empty`) and prints
`self-test OK` (exit 0) or a list of failed assertions (exit 1).

To sanity-check `run_docker_review.sh` itself without running a real review:

```bash
bash -n e2e/run_docker_review.sh
sh -n e2e/run_docker_review.sh
e2e/run_docker_review.sh --help
```
