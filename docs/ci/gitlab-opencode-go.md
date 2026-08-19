# 📘 GitLab CI/CD with the OpenCode Go gateway

## ⚙️ Settings

The OpenCode Go gateway (`https://opencode.ai/zen/go/v1`) serves different models on different endpoints: some
models are only available on `/chat/completions`, others only on `/responses`. Use `LLM__META__API` to pin the
endpoint per model instead of relying on the automatic model-prefix detection.

- `LLM__META__API` — `AUTO` (default, prefix-based detection) | `CHAT` (force `/chat/completions`) | `RESPONSES`
  (force `/responses`)
- `LLM__HTTP_CLIENT__CONNECT_TIMEOUT` — TCP connect timeout in seconds, separate from the request `timeout`
- `LLM__META__EXTRA_BODY` — JSON object merged into the request body; use it to disable reasoning/thinking on
  models that default to spending a large, slow reasoning budget on every step

Example:

```yaml
LLM__HTTP_CLIENT__API_URL: "https://opencode.ai/zen/go/v1"
LLM__HTTP_CLIENT__CONNECT_TIMEOUT: "5"
LLM__META__MODEL: "deepseek-v4-flash"
LLM__META__API: "CHAT"
LLM__META__EXTRA_BODY: '{"thinking":{"type":"disabled"}}'
```

```yaml
LLM__HTTP_CLIENT__API_URL: "https://opencode.ai/zen/go/v1"
LLM__META__MODEL: "grok-4.5"
LLM__META__API: "RESPONSES"
LLM__META__EXTRA_BODY: '{"reasoning":{"effort":"low"}}'
```

## 🧭 Why this route

- The Go plan base URL is `https://opencode.ai/zen/go/v1` (responses carry `"cost":"0"`); `https://opencode.ai/zen/v1`
  spends pay-as-you-go credits with the same key.
- `deepseek-v4-flash` is served by `/chat/completions` (`LLM__META__API=CHAT`, streaming optional and off by
  default); `grok-4.5` is served ONLY by `/responses` (`LLM__META__API=RESPONSES`). Streaming is supported on both
  endpoints; see the grok job note below for why it's turned on there specifically.
- One image serves both jobs.

## ⏱️ Time budget arithmetic

```
job timeout >= AGENT__DEADLINE_SECONDS + LLM__HTTP_CLIENT__TIMEOUT * (2 + AGENT__FORCE_FINAL_ATTEMPTS) + 300s
```

Recommended: `LLM__HTTP_CLIENT__TIMEOUT=300`, `AGENT__DEADLINE_SECONDS=600`, `AGENT__FORCE_FINAL_ATTEMPTS=2` → hard
cap 2100s. Set the GitLab job `timeout:` to **35 minutes** for the automatic (deepseek) job, and **40 minutes** for
the grok job — the extra headroom accounts for `/responses` not supporting streaming.

This formula assumes no transport retries. `RetryTransport` retries 5xx responses up to 5 times per request, so a
gateway brown-out can push a single call well past `LLM__HTTP_CLIENT__TIMEOUT` and the whole job past the hard cap
above — treat the GitLab job `timeout:` as the actual hard stop, not this formula (`allow_failure: true` is what
keeps a timeout advisory rather than blocking).

## 🧱 Shared job template

```yaml
.ai_review_base:
  stage: review
  tags: [app-expert, android, debian]
  only: [merge_requests]
  allow_failure: true
  timeout: 35 minutes
  before_script: []
  variables:
    GIT_DEPTH: 30
    GIT_SUBMODULE_STRATEGY: none
    AI_REVIEW_IMAGE: ghcr.io/pablodiguerero/ai-review:<branch-with-dashes>-<commit-sha>
    AI_REVIEW_MODEL: deepseek-v4-flash
    AI_REVIEW_API: CHAT
    AI_REVIEW_STREAM: "false"
    AI_REVIEW_TIMEOUT: "300"
    AI_REVIEW_EXTRA_BODY: '{"thinking":{"type":"disabled"}}'
    AI_REVIEW_SUMMARY_TAG: "#ai-review-summary"
    AI_REVIEW_SUMMARY_REPLY_TAG: "#ai-review-reply"
    AI_REVIEW_FAIL_ON_EMPTY: "false"
    AI_REVIEW_SUMMARY_HEADER: "### AI review: {model}"
  script: |
    if [ -n "$CI_MERGE_REQUEST_DIFF_BASE_SHA" ]; then
      git fetch --no-tags --depth=1 origin "$CI_MERGE_REQUEST_DIFF_BASE_SHA" 2>/dev/null || true
    fi
    export LLM__HTTP_CLIENT__API_TOKEN="$OPENCODE_ZEN_API_KEY"
    export VCS__HTTP_CLIENT__API_TOKEN="$AI_REVIEW_GITLAB_TOKEN"
    docker run --rm --user $(id -u):$(id -g) \
      -v $(pwd):/app -w /app \
      -e LLM__HTTP_CLIENT__API_TOKEN -e VCS__HTTP_CLIENT__API_TOKEN \
      -e AI_REVIEW_CONFIG_FILE_YAML=/nonexistent -e AI_REVIEW_CONFIG_FILE_JSON=/nonexistent -e AI_REVIEW_CONFIG_FILE_ENV=/nonexistent \
      -e LLM__PROVIDER="OPENAI" \
      -e LLM__META__MODEL="$AI_REVIEW_MODEL" \
      -e LLM__META__API="$AI_REVIEW_API" \
      -e LLM__META__STREAM="$AI_REVIEW_STREAM" \
      -e LLM__META__MAX_TOKENS="32000" \
      -e LLM__META__TEMPERATURE="0.3" \
      -e LLM__META__EXTRA_BODY="$AI_REVIEW_EXTRA_BODY" \
      -e LLM__HTTP_CLIENT__API_URL="https://opencode.ai/zen/go/v1" \
      -e LLM__HTTP_CLIENT__TIMEOUT="$AI_REVIEW_TIMEOUT" \
      -e LLM__HTTP_CLIENT__CONNECT_TIMEOUT="10" \
      -e VCS__PROVIDER="GITLAB" \
      -e VCS__PIPELINE__PROJECT_ID="$CI_PROJECT_ID" \
      -e VCS__PIPELINE__MERGE_REQUEST_ID="$CI_MERGE_REQUEST_IID" \
      -e VCS__HTTP_CLIENT__API_URL="$CI_SERVER_URL" \
      -e REVIEW__MODE="FULL_FILE_DIFF" \
      -e REVIEW__SUMMARY_TAG="$AI_REVIEW_SUMMARY_TAG" \
      -e REVIEW__SUMMARY_REPLY_TAG="$AI_REVIEW_SUMMARY_REPLY_TAG" \
      -e REVIEW__SUMMARY_FEEDBACK_LOOP="true" \
      -e REVIEW__FAIL_ON_EMPTY_RESULT="$AI_REVIEW_FAIL_ON_EMPTY" \
      -e REVIEW__SUMMARY_HEADER="$AI_REVIEW_SUMMARY_HEADER" \
      -e REVIEW__SUMMARY_REPLACE_PREVIOUS="true" \
      -e PROMPT__SUMMARY_PROMPT_FILES='["./.ai-review/prompts/summary.md", "./.ai-review/prompts/verify.md"]' \
      -e AGENT__ENABLED="true" \
      -e AGENT__MAX_ITERATIONS="30" \
      -e AGENT__MIN_TOOL_CALLS="2" \
      -e AGENT__DEADLINE_SECONDS="600" \
      -e AGENT__FORCE_FINAL_ATTEMPTS="2" \
      --entrypoint sh "$AI_REVIEW_IMAGE" -c "ai-review run-summary"

ai_review_job:
  extends: .ai_review_base

ai_review_grok_job:
  extends: .ai_review_base
  when: manual
  timeout: 40 minutes
  variables:
    AI_REVIEW_MODEL: grok-4.5
    AI_REVIEW_API: RESPONSES
    AI_REVIEW_STREAM: "true"
    AI_REVIEW_TIMEOUT: "150"
    AI_REVIEW_EXTRA_BODY: '{}'
    AI_REVIEW_SUMMARY_TAG: "#ai-review-grok-summary"
    AI_REVIEW_SUMMARY_REPLY_TAG: "#ai-review-grok-reply"
```

## 🔒 Security

AI Review loads `.ai-review.yaml` / `.ai-review.json` / `.env` from its working directory by default, and those
files take priority over environment variables. In this job the working directory is the reviewed checkout
(`-v $(pwd):/app -w /app`), so without the `AI_REVIEW_CONFIG_FILE_*` pins above, any merge request could add or
edit one of those files and reconfigure the reviewer itself — for example widening the agent's command allowlist,
or pointing `LLM__HTTP_CLIENT__API_URL`/`VCS__HTTP_CLIENT__API_URL` at an attacker-controlled host to exfiltrate the
job's tokens. Pointing all three `AI_REVIEW_CONFIG_FILE_*` variables at a path that doesn't exist neutralises this:
a missing config file is skipped, so only the `-e` variables above can configure the run (verified against this
codebase's config loader).

Even pinned, the agent's read-only shell commands (`cat`, `grep`, `find`, …) can read anything the container user
can read, which includes the whole checkout. `/proc/self/environ` is blocked, but a repo file with a plaintext
secret committed in it is still read and sent to the LLM gateway like any other file. Keep the CI job token and
`OPENCODE_ZEN_API_KEY` short-lived/scoped, and rotate any secret that ends up committed in a reviewed repository.

## 📝 Notes

- `AI_REVIEW_STREAM="true"` + `AI_REVIEW_TIMEOUT="150"` on the grok job: the first `/responses` request to
  grok-4.5 on this gateway frequently stalls with no bytes at all until the read timeout, while a repeated request
  usually answers in seconds. Streaming lets the client detect a stall (no SSE bytes within `LLM__HTTP_CLIENT__TIMEOUT`)
  and retry once at zero cost, since nothing billable was generated before the drop — so a shorter timeout here
  means faster stall detection, not a smaller answer budget. deepseek stays non-streaming (`AI_REVIEW_STREAM="false"`,
  `AI_REVIEW_TIMEOUT="300"`) since it isn't affected by this stall. With `AI_REVIEW_TIMEOUT=150` the "Time budget
  arithmetic" formula above gives grok a smaller hard cap than the 40-minute job `timeout:`, which stays generous
  on purpose.
- `OPENCODE_ZEN_API_KEY` = the Go plan key (project CI variable, masked). `DEEPSEEK_API_KEY` is no longer read.
- `AI_REVIEW_EXTRA_BODY='{"thinking":{"type":"disabled"}}'` on the deepseek job cuts reasoning tokens and per-step
  time roughly 10x on this gateway; even with thinking disabled, reviews still run slower during DeepSeek's peak
  hours (06:00–10:00 UTC).
- The GHCR image tag is now the branch name with `/` replaced by `-`, plus the commit sha (e.g.
  `feat-openai-stream-a1b2c3d`); the old `:agent-json-mode` tag is no longer updated.
- expert-app: keep its `--env-file ai-review.env` + `umask 077` secret passing and its `yarn install` step; set
  `AI_REVIEW_FAIL_ON_EMPTY: "true"` so a gateway outage shows as a red advisory job (the merge gate stays
  fail-closed; unblock = approve + retry job).
- cabinet: replace `/zen/v1` on server/master and fix/report-api3-empty-boolean; on fix/ai-review-go-plan-route drop
  `.ai-review/opencode.Dockerfile` and `.ai-review/opencode_review.sh` (grok now runs through the same image with
  `LLM__META__API=RESPONSES`); update CLAUDE.md lines claiming the gateway needs `stream: true`.
- client-app: new job; copy `.ai-review/prompts` from expert-app and adapt, or run with default prompts (drop
  `PROMPT__SUMMARY_PROMPT_FILES`).
- Comment tag lookup matches a standalone line of the comment body, not a substring, so `#ai-review-grok-summary`
  and `#ai-review-summary` never cross-match even though one name contains the other.
- Do not use `*-free` models. Rotate plaintext credentials that live in the reviewed repositories: the agent sends
  everything it reads to the gateway.
- Fallback during a gateway outage: `LLM__PROVIDER=OPENROUTER` with the OpenRouter key and model id (config-only
  change).
- To let a retried job skip tool iterations it already paid for, mount a runner-persistent directory and set
  `AGENT__CHECKPOINT_DIR`, e.g. add `-v "$HOME/.ai-review-cache:/cache"` to the `docker run` line and
  `-e AGENT__CHECKPOINT_DIR=/cache`. This only helps on a shell runner where `$HOME` is stable across job retries
  on the same host (a fresh Docker/Kubernetes executor gets an empty volume every time). The checkpoint key
  includes `head_sha`, so a new push always starts a clean loop — only a retry of the *same* commit resumes.
