#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: e2e/run_docker_review.sh [--help]

Runs the ai-review docker image against a real GitLab merge request in
dry-run mode, exactly like a consumer CI job would, then asserts the
result from the produced JSON artifacts.

Everything is configured through environment variables.

Required:
  E2E_NAME               short label for the repo clone and run directory
  E2E_PROJECT_ID         VCS__PIPELINE__PROJECT_ID
  E2E_MR_IID             VCS__PIPELINE__MERGE_REQUEST_ID
  E2E_MODEL              LLM__META__MODEL
  OPENCODE_API_KEY       LLM__HTTP_CLIENT__API_TOKEN (secret)
  AI_REVIEW_GITLAB_TOKEN VCS__HTTP_CLIENT__API_TOKEN (secret)
  one of E2E_REPO_LOCAL / E2E_REPO_URL, unless the clone already exists

Workspace and repo:
  E2E_WORKSPACE           default: $HOME/tmp/ai-review-e2e (must live under $HOME for colima bind mounts)
  E2E_REPO_LOCAL          local path to clone from (fast, hardlinked)
  E2E_REPO_URL            git remote to point origin at / fetch from
  E2E_PREPARE             command run inside the clone before docker (e.g. yarn install)

Docker image:
  E2E_IMAGE               default: ai-review:e2e
  E2E_BUILD               1 to docker build the image from the repo root first
  E2E_EXTRA_DOCKER_ARGS   extra args appended verbatim (word-split) to docker run

GitLab:
  E2E_GITLAB_API_URL      default: https://gitlab.com

LLM:
  E2E_API                 LLM__META__API: AUTO|CHAT|RESPONSES, default AUTO
  E2E_STREAM              LLM__META__STREAM, default false
  E2E_MAX_TOKENS          LLM__META__MAX_TOKENS, default 32000
  E2E_TEMPERATURE         LLM__META__TEMPERATURE, default 0.3
  E2E_LLM_API_URL         LLM__HTTP_CLIENT__API_URL, default https://opencode.ai/zen/go/v1
  E2E_LLM_TIMEOUT         LLM__HTTP_CLIENT__TIMEOUT, default 240
  E2E_LLM_CONNECT_TIMEOUT LLM__HTTP_CLIENT__CONNECT_TIMEOUT, default 10

Review:
  E2E_REVIEW_MODE             REVIEW__MODE, default FULL_FILE_DIFF
  E2E_SUMMARY_FEEDBACK_LOOP   REVIEW__SUMMARY_FEEDBACK_LOOP, default true
  E2E_FAIL_ON_EMPTY_RESULT    REVIEW__FAIL_ON_EMPTY_RESULT, only passed through when set
  E2E_PROMPT_FILES            PROMPT__SUMMARY_PROMPT_FILES, JSON list string; unset means built-in defaults

Agent:
  E2E_AGENT_ENABLED             AGENT__ENABLED, default true
  E2E_AGENT_MAX_ITERATIONS      AGENT__MAX_ITERATIONS, default 30
  E2E_AGENT_MIN_TOOL_CALLS      AGENT__MIN_TOOL_CALLS, default 2
  E2E_AGENT_DEADLINE_SECONDS    AGENT__DEADLINE_SECONDS, default 300
  E2E_AGENT_FORCE_FINAL_ATTEMPTS AGENT__FORCE_FINAL_ATTEMPTS, default 1

Assertion (forwarded to assert_review.py):
  E2E_EXPECT_SCORE        1 to require an "Overall score: N" line
  E2E_MIN_EXECUTED         minimum data.agent.executed_tool_calls
  E2E_EXPECT_EXIT          expected docker exit code, default 0
  E2E_ALLOW_TRACEBACK      1 to tolerate a Traceback in run.log
  E2E_EXPECT_EMPTY         1 to require an empty final response instead
  E2E_ASSERT_JSON          1 to print assert_review.py metrics as JSON

Misc:
  E2E_PYTHON               python interpreter for assert_review.py, default python3.12 (falls back to python3)

Output lands under $E2E_WORKSPACE/runs/<name>-mr<iid>-<model>-<timestamp>/,
containing run.log, artifacts/llm, artifacts/vcs, env.txt and meta.json.
The script exits with assert_review.py's result, not the docker exit code.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

log_info() {
  printf '[run-docker-review] %s\n' "$1" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    fail "missing required environment variable: $name"
  fi
}

resolve_python() {
  local candidate="${E2E_PYTHON:-python3.12}"
  if ! command -v "$candidate" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      log_info "E2E_PYTHON candidate '$candidate' not found, falling back to python3"
      candidate="python3"
    else
      fail "no python interpreter found; set E2E_PYTHON to a Python >=3.10 binary"
    fi
  fi
  if ! "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    fail "$candidate is older than Python 3.10; set E2E_PYTHON to a newer interpreter"
  fi
  PYTHON="$candidate"
}

fetch_mr_metadata() {
  local url config response
  url="${gitlab_api_url%/}/api/v4/projects/${project_id}/merge_requests/${mr_iid}"
  config="$(mktemp)"
  chmod 600 "$config"
  printf 'header = "PRIVATE-TOKEN: %s"\n' "$AI_REVIEW_GITLAB_TOKEN" > "$config"

  if ! response="$(curl --fail --silent --show-error -K "$config" "$url")"; then
    rm -f "$config"
    fail "failed to fetch merge request metadata from $url"
  fi
  rm -f "$config"

  export MR_RESPONSE="$response"
  eval "$("$PYTHON" <<'PYEOF'
import json
import os
import shlex

data = json.loads(os.environ.get("MR_RESPONSE", ""))
diff_refs = data.get("diff_refs") or {}
fields = {
    "MR_BASE_SHA": diff_refs.get("base_sha") or "",
    "MR_HEAD_SHA": diff_refs.get("head_sha") or "",
    "MR_SOURCE_BRANCH": data.get("source_branch") or "",
    "MR_CHANGES_COUNT": str(data.get("changes_count") or ""),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(value)}")
PYEOF
)"
  unset MR_RESPONSE

  [ -n "${MR_BASE_SHA:-}" ] || fail "GitLab response missing diff_refs.base_sha"
  [ -n "${MR_HEAD_SHA:-}" ] || fail "GitLab response missing diff_refs.head_sha"
}

clone_or_update_repo() {
  repo_dir="${workspace}/${name}"

  if [ ! -d "${repo_dir}/.git" ]; then
    local clone_source="${E2E_REPO_LOCAL:-${E2E_REPO_URL:-}}"
    [ -n "$clone_source" ] || fail "repo clone missing at ${repo_dir} and neither E2E_REPO_LOCAL nor E2E_REPO_URL is set"
    mkdir -p "$workspace"
    log_info "cloning ${clone_source} into ${repo_dir}"
    git clone --no-checkout "$clone_source" "$repo_dir"
  fi

  if [ -n "${E2E_REPO_URL:-}" ]; then
    git -C "$repo_dir" remote set-url origin "$E2E_REPO_URL"
  fi

  log_info "fetching refs/merge-requests/${mr_iid}/head"
  git -C "$repo_dir" fetch origin "refs/merge-requests/${mr_iid}/head"
  git -C "$repo_dir" fetch origin "$MR_BASE_SHA" || true

  git -C "$repo_dir" checkout --detach "$MR_HEAD_SHA"

  if ! git -C "$repo_dir" cat-file -t "$MR_BASE_SHA" >/dev/null 2>&1; then
    fail "base_sha ${MR_BASE_SHA} is not present locally after fetch; the server may forbid direct sha fetches"
  fi
}

run_prepare() {
  if [ -n "${E2E_PREPARE:-}" ]; then
    log_info "running E2E_PREPARE in ${repo_dir}"
    (cd "$repo_dir" && eval "$E2E_PREPARE")
  fi
}

build_image_if_requested() {
  if [ "${E2E_BUILD:-0}" = "1" ]; then
    local repo_root
    repo_root="$(cd "${script_dir}/.." && pwd)"
    log_info "building ${image} from ${repo_root}"
    docker build -t "$image" "$repo_root"
  fi
}

write_env_txt() {
  printf '%s\n' "${env_summary[@]}" > "${run_dir}/env.txt"
}

run_docker_review() {
  local docker_user
  docker_user="$(id -u):$(id -g)"

  docker_args=(
    run --rm
    --user "$docker_user"
    -v "${repo_dir}:/app"
    -w /app
    -v "${run_dir}/artifacts:/artifacts"
    -e LLM__PROVIDER=OPENAI
    -e "LLM__META__MODEL=${model}"
    -e "LLM__META__API=${api}"
    -e "LLM__META__STREAM=${stream}"
    -e "LLM__META__MAX_TOKENS=${max_tokens}"
    -e "LLM__META__TEMPERATURE=${temperature}"
    -e "LLM__HTTP_CLIENT__API_URL=${llm_api_url}"
    -e LLM__HTTP_CLIENT__API_TOKEN
    -e "LLM__HTTP_CLIENT__TIMEOUT=${llm_timeout}"
    -e "LLM__HTTP_CLIENT__CONNECT_TIMEOUT=${llm_connect_timeout}"
    -e VCS__PROVIDER=GITLAB
    -e "VCS__PIPELINE__PROJECT_ID=${project_id}"
    -e "VCS__PIPELINE__MERGE_REQUEST_ID=${mr_iid}"
    -e "VCS__HTTP_CLIENT__API_URL=${gitlab_api_url}"
    -e VCS__HTTP_CLIENT__API_TOKEN
    -e "REVIEW__MODE=${review_mode}"
    -e REVIEW__DRY_RUN=true
    -e "REVIEW__SUMMARY_FEEDBACK_LOOP=${summary_feedback_loop}"
    -e "AGENT__ENABLED=${agent_enabled}"
    -e "AGENT__MAX_ITERATIONS=${agent_max_iterations}"
    -e "AGENT__MIN_TOOL_CALLS=${agent_min_tool_calls}"
    -e "AGENT__DEADLINE_SECONDS=${agent_deadline_seconds}"
    -e "AGENT__FORCE_FINAL_ATTEMPTS=${agent_force_final_attempts}"
    -e ARTIFACTS__LLM_ENABLED=true
    -e ARTIFACTS__VCS_ENABLED=true
    -e ARTIFACTS__LLM_DIR=/artifacts/llm
    -e ARTIFACTS__VCS_DIR=/artifacts/vcs
    -e AI_REVIEW_CONFIG_FILE_YAML=/nonexistent
    -e AI_REVIEW_CONFIG_FILE_JSON=/nonexistent
    -e AI_REVIEW_CONFIG_FILE_ENV=/nonexistent
    -e PYTHONUNBUFFERED=1
  )

  env_summary=(
    "LLM__PROVIDER=OPENAI"
    "LLM__META__MODEL=${model}"
    "LLM__META__API=${api}"
    "LLM__META__STREAM=${stream}"
    "LLM__META__MAX_TOKENS=${max_tokens}"
    "LLM__META__TEMPERATURE=${temperature}"
    "LLM__HTTP_CLIENT__API_URL=${llm_api_url}"
    "LLM__HTTP_CLIENT__API_TOKEN=<redacted>"
    "LLM__HTTP_CLIENT__TIMEOUT=${llm_timeout}"
    "LLM__HTTP_CLIENT__CONNECT_TIMEOUT=${llm_connect_timeout}"
    "VCS__PROVIDER=GITLAB"
    "VCS__PIPELINE__PROJECT_ID=${project_id}"
    "VCS__PIPELINE__MERGE_REQUEST_ID=${mr_iid}"
    "VCS__HTTP_CLIENT__API_URL=${gitlab_api_url}"
    "VCS__HTTP_CLIENT__API_TOKEN=<redacted>"
    "REVIEW__MODE=${review_mode}"
    "REVIEW__DRY_RUN=true"
    "REVIEW__SUMMARY_FEEDBACK_LOOP=${summary_feedback_loop}"
    "AGENT__ENABLED=${agent_enabled}"
    "AGENT__MAX_ITERATIONS=${agent_max_iterations}"
    "AGENT__MIN_TOOL_CALLS=${agent_min_tool_calls}"
    "AGENT__DEADLINE_SECONDS=${agent_deadline_seconds}"
    "AGENT__FORCE_FINAL_ATTEMPTS=${agent_force_final_attempts}"
    "ARTIFACTS__LLM_ENABLED=true"
    "ARTIFACTS__VCS_ENABLED=true"
    "ARTIFACTS__LLM_DIR=/artifacts/llm"
    "ARTIFACTS__VCS_DIR=/artifacts/vcs"
    "AI_REVIEW_CONFIG_FILE_YAML=/nonexistent"
    "AI_REVIEW_CONFIG_FILE_JSON=/nonexistent"
    "AI_REVIEW_CONFIG_FILE_ENV=/nonexistent"
    "PYTHONUNBUFFERED=1"
  )

  if [ -n "${E2E_FAIL_ON_EMPTY_RESULT:-}" ]; then
    docker_args+=( -e "REVIEW__FAIL_ON_EMPTY_RESULT=${E2E_FAIL_ON_EMPTY_RESULT}" )
    env_summary+=( "REVIEW__FAIL_ON_EMPTY_RESULT=${E2E_FAIL_ON_EMPTY_RESULT}" )
  fi

  if [ -n "${E2E_PROMPT_FILES:-}" ]; then
    docker_args+=( -e "PROMPT__SUMMARY_PROMPT_FILES=${E2E_PROMPT_FILES}" )
    env_summary+=( "PROMPT__SUMMARY_PROMPT_FILES=${E2E_PROMPT_FILES}" )
  fi

  if [ -n "${E2E_EXTRA_DOCKER_ARGS:-}" ]; then
    set -f
    docker_args+=( $E2E_EXTRA_DOCKER_ARGS )
    set +f
    env_summary+=( "E2E_EXTRA_DOCKER_ARGS=${E2E_EXTRA_DOCKER_ARGS}" )
  fi

  docker_args+=( --entrypoint sh "$image" -c "ai-review run-summary" )

  write_env_txt

  export LLM__HTTP_CLIENT__API_TOKEN="$OPENCODE_API_KEY"
  export VCS__HTTP_CLIENT__API_TOKEN="$AI_REVIEW_GITLAB_TOKEN"

  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log_info "running docker review, output at ${run_dir}/run.log"

  set +e
  docker "${docker_args[@]}" 2>&1 | tee "${run_dir}/run.log"
  exit_code="${PIPESTATUS[0]}"
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log_info "docker exited with code ${exit_code}"
}

write_meta_json() {
  export META_NAME="$name" \
         META_PROJECT_ID="$project_id" \
         META_MR_IID="$mr_iid" \
         META_BASE_SHA="$MR_BASE_SHA" \
         META_HEAD_SHA="$MR_HEAD_SHA" \
         META_SOURCE_BRANCH="$MR_SOURCE_BRANCH" \
         META_CHANGES_COUNT="$MR_CHANGES_COUNT" \
         META_MODEL="$model" \
         META_API="$api" \
         META_STREAM="$stream" \
         META_IMAGE="$image" \
         META_REVIEW_MODE="$review_mode" \
         META_STARTED_AT="$started_at" \
         META_FINISHED_AT="$finished_at" \
         META_DOCKER_EXIT_CODE="$exit_code" \
         META_RUN_DIR="$run_dir"

  "$PYTHON" <<'PYEOF'
import json
import os

def env(name):
    return os.environ.get(name, "")

data = {
    "name": env("META_NAME"),
    "project_id": env("META_PROJECT_ID"),
    "mr_iid": env("META_MR_IID"),
    "base_sha": env("META_BASE_SHA"),
    "head_sha": env("META_HEAD_SHA"),
    "source_branch": env("META_SOURCE_BRANCH"),
    "changes_count": env("META_CHANGES_COUNT"),
    "model": env("META_MODEL"),
    "api": env("META_API"),
    "stream": env("META_STREAM") == "true",
    "image": env("META_IMAGE"),
    "review_mode": env("META_REVIEW_MODE"),
    "started_at": env("META_STARTED_AT"),
    "finished_at": env("META_FINISHED_AT"),
    "docker_exit_code": int(env("META_DOCKER_EXIT_CODE") or 1),
}

with open(os.path.join(env("META_RUN_DIR"), "meta.json"), "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PYEOF
}

run_assert() {
  local assert_args=(--run-dir "$run_dir" --expect-exit "${E2E_EXPECT_EXIT:-0}")

  if [ "${E2E_EXPECT_SCORE:-0}" = "1" ]; then
    assert_args+=(--expect-score)
  fi
  if [ -n "${E2E_MIN_EXECUTED:-}" ]; then
    assert_args+=(--min-executed "$E2E_MIN_EXECUTED")
  fi
  if [ "${E2E_ALLOW_TRACEBACK:-0}" = "1" ]; then
    assert_args+=(--allow-traceback)
  fi
  if [ "${E2E_EXPECT_EMPTY:-0}" = "1" ]; then
    assert_args+=(--expect-empty)
  fi
  if [ "${E2E_ASSERT_JSON:-0}" = "1" ]; then
    assert_args+=(--json)
  fi

  "$PYTHON" "${script_dir}/assert_review.py" "${assert_args[@]}"
}

main() {
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        usage
        exit 0
        ;;
      *)
        usage
        fail "unknown argument: $arg"
        ;;
    esac
  done

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  require_cmd docker
  require_cmd git
  require_cmd curl
  require_cmd mktemp

  require_env E2E_NAME
  require_env E2E_PROJECT_ID
  require_env E2E_MR_IID
  require_env E2E_MODEL
  require_env OPENCODE_API_KEY
  require_env AI_REVIEW_GITLAB_TOKEN

  resolve_python

  name="$E2E_NAME"
  workspace="${E2E_WORKSPACE:-$HOME/tmp/ai-review-e2e}"
  case "$workspace" in
    "$HOME"/*) ;;
    *) fail "E2E_WORKSPACE must live under \$HOME for colima bind mounts to work, got: $workspace" ;;
  esac

  project_id="$E2E_PROJECT_ID"
  mr_iid="$E2E_MR_IID"
  gitlab_api_url="${E2E_GITLAB_API_URL:-https://gitlab.com}"

  model="$E2E_MODEL"
  api="${E2E_API:-AUTO}"
  stream="${E2E_STREAM:-false}"
  max_tokens="${E2E_MAX_TOKENS:-32000}"
  temperature="${E2E_TEMPERATURE:-0.3}"
  llm_api_url="${E2E_LLM_API_URL:-https://opencode.ai/zen/go/v1}"
  llm_timeout="${E2E_LLM_TIMEOUT:-240}"
  llm_connect_timeout="${E2E_LLM_CONNECT_TIMEOUT:-10}"

  review_mode="${E2E_REVIEW_MODE:-FULL_FILE_DIFF}"
  summary_feedback_loop="${E2E_SUMMARY_FEEDBACK_LOOP:-true}"

  agent_enabled="${E2E_AGENT_ENABLED:-true}"
  agent_max_iterations="${E2E_AGENT_MAX_ITERATIONS:-30}"
  agent_min_tool_calls="${E2E_AGENT_MIN_TOOL_CALLS:-2}"
  agent_deadline_seconds="${E2E_AGENT_DEADLINE_SECONDS:-300}"
  agent_force_final_attempts="${E2E_AGENT_FORCE_FINAL_ATTEMPTS:-1}"

  image="${E2E_IMAGE:-ai-review:e2e}"

  build_image_if_requested
  fetch_mr_metadata

  clone_or_update_repo
  run_prepare

  local safe_model="${model//\//_}"
  local run_ts
  run_ts="$(date -u +%Y%m%dT%H%M%SZ)"
  run_dir="${workspace}/runs/${name}-mr${mr_iid}-${safe_model}-${run_ts}"
  mkdir -p "${run_dir}/artifacts/llm" "${run_dir}/artifacts/vcs"

  run_docker_review
  write_meta_json

  run_assert
}

main "$@"
