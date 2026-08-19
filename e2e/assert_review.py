import argparse
import json
import re
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

PROTOCOL_LEAK_MARKERS = ("<TOOL_CALL>", '"action":', '"TOOL_CALL"', "</TOOL_CALL>")
TRACEBACK_MARKER = "Traceback (most recent call last)"
AGENT_LOOP_SUMMARY_RE = re.compile(
    r"Agent loop summary: iterations=(\d+) executed_tool_calls=(\d+) blocked_tool_calls=(\d+) stop_reason=(\S+)"
)
FORCED_OR_DEADLINE_RE = re.compile(r"forced|deadline", re.IGNORECASE)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json_file(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def iter_artifact_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def load_llm_artifacts(run_dir: Path) -> list[dict]:
    directory = run_dir / "artifacts" / "llm"
    artifacts = []
    for file_path in iter_artifact_files(directory):
        data = load_json_file(file_path)
        if isinstance(data, dict):
            artifacts.append(data)
    return sorted(artifacts, key=lambda item: item.get("timestamp") or "")


def load_vcs_summary_artifacts(run_dir: Path) -> list[dict]:
    directory = run_dir / "artifacts" / "vcs"
    artifacts = []
    for file_path in iter_artifact_files(directory):
        data = load_json_file(file_path)
        if isinstance(data, dict) and data.get("type") == "VCS_SUMMARY":
            artifacts.append(data)
    return sorted(artifacts, key=lambda item: item.get("timestamp") or "")


def parse_iso(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def check_exit_code(meta: dict, expect_exit: int) -> tuple[bool, str]:
    actual = meta.get("docker_exit_code")
    if actual is None:
        return False, "docker_exit_code missing from meta.json"
    if actual == expect_exit:
        return True, f"docker exit code {actual} == expected {expect_exit}"
    return False, f"docker exit code {actual} != expected {expect_exit}"


def check_no_traceback(log_text: str, allow_traceback: bool) -> tuple[bool, str]:
    if allow_traceback:
        return True, "traceback check skipped (--allow-traceback)"
    if TRACEBACK_MARKER in log_text:
        return False, f"found '{TRACEBACK_MARKER}' in run.log"
    return True, "no traceback found in run.log"


def check_llm_response(
        llm_artifacts: list[dict],
        vcs_summary_artifacts: list[dict],
        expect_empty: bool,
) -> tuple[bool, str, dict | None]:
    last_artifact = llm_artifacts[-1] if llm_artifacts else None
    response = (last_artifact.get("data") or {}).get("response") or "" if last_artifact else ""

    if expect_empty:
        if last_artifact is not None and response.strip():
            return False, "expected empty response but last LLM artifact has content", last_artifact
        if vcs_summary_artifacts:
            return False, "expected no VCS summary artifact but one exists", last_artifact
        if last_artifact is None:
            return True, "no LLM artifacts and no VCS summary artifact, as expected", last_artifact
        return True, "response is empty and no VCS summary artifact, as expected", last_artifact

    if last_artifact is None:
        return False, "no LLM artifacts found", None

    if not response.strip():
        return False, "last LLM artifact has an empty response", last_artifact
    return True, f"last LLM artifact has a {len(response)}-char response", last_artifact


def check_no_protocol_leak(response: str) -> tuple[bool, str]:
    hits = [marker for marker in PROTOCOL_LEAK_MARKERS if marker in response]
    if hits:
        return False, f"response leaks protocol markers: {hits}"
    return True, "no protocol markers leaked in response"


def check_score(response: str) -> tuple[bool, str]:
    if re.search(r"Overall score:\s*\d", response):
        return True, "response contains an 'Overall score: <n>' line"
    return False, "response missing 'Overall score: <n>' line"


def agent_stats_from_artifact(last_artifact: dict | None) -> dict | None:
    data = (last_artifact or {}).get("data") or {}
    agent = data.get("agent")
    return agent if isinstance(agent, dict) else None


def agent_stats_from_log(log_text: str) -> dict | None:
    match = AGENT_LOOP_SUMMARY_RE.search(log_text)
    if not match:
        return None
    return {
        "iterations": int(match.group(1)),
        "executed_tool_calls": int(match.group(2)),
        "blocked_tool_calls": int(match.group(3)),
        "stop_reason": match.group(4),
    }


def resolve_agent_stats(last_artifact: dict | None, log_text: str) -> dict | None:
    return agent_stats_from_artifact(last_artifact) or agent_stats_from_log(log_text)


def check_min_executed(last_artifact: dict | None, log_text: str, minimum: int) -> tuple[bool, str]:
    stats = resolve_agent_stats(last_artifact, log_text)
    executed = stats.get("executed_tool_calls") if stats else None
    if executed is None:
        return False, "agent stats unavailable"
    if executed >= minimum:
        return True, f"executed_tool_calls {executed} >= {minimum}"
    return False, f"executed_tool_calls {executed} < {minimum}"


def compute_metrics(meta: dict, log_text: str, llm_artifacts: list[dict], last_artifact: dict | None) -> dict:
    first_artifact = llm_artifacts[0] if llm_artifacts else None
    first_prompt_chars = len((first_artifact.get("data") or {}).get("prompt") or "") if first_artifact else 0
    last_response_chars = len((last_artifact.get("data") or {}).get("response") or "") if last_artifact else 0

    stats = resolve_agent_stats(last_artifact, log_text) or {}
    stop_reason = stats.get("stop_reason")

    started = parse_iso(meta.get("started_at"))
    finished = parse_iso(meta.get("finished_at"))
    wall_seconds = (finished - started).total_seconds() if started and finished else None

    return {
        "llm_artifact_count": len(llm_artifacts),
        "first_prompt_chars": first_prompt_chars,
        "last_response_chars": last_response_chars,
        "iterations": stats.get("iterations"),
        "executed_tool_calls": stats.get("executed_tool_calls"),
        "blocked_tool_calls": stats.get("blocked_tool_calls"),
        "stop_reason": stop_reason,
        "wall_seconds": wall_seconds,
        "waiting_for_response_lines": log_text.count("Waiting for response"),
        "blocked_by_policy_lines": log_text.count("blocked by policy"),
        "stop_reason_forced_or_deadline": bool(stop_reason and FORCED_OR_DEADLINE_RE.search(stop_reason)),
    }


def evaluate(
        run_dir: Path,
        expect_score: bool,
        min_executed: int | None,
        expect_exit: int,
        allow_traceback: bool,
        expect_empty: bool,
) -> tuple[list[dict], dict, bool]:
    meta = load_json_file(run_dir / "meta.json") or {}
    log_text = read_text(run_dir / "run.log")
    llm_artifacts = load_llm_artifacts(run_dir)
    vcs_summary_artifacts = load_vcs_summary_artifacts(run_dir)

    checks = []

    passed, reason = check_exit_code(meta, expect_exit)
    checks.append({"name": "exit_code", "passed": passed, "reason": reason})

    passed, reason = check_no_traceback(log_text, allow_traceback)
    checks.append({"name": "no_traceback", "passed": passed, "reason": reason})

    passed, reason, last_artifact = check_llm_response(llm_artifacts, vcs_summary_artifacts, expect_empty)
    checks.append({"name": "llm_response", "passed": passed, "reason": reason})

    response_text = (last_artifact.get("data") or {}).get("response") or "" if last_artifact else ""

    passed, reason = check_no_protocol_leak(response_text)
    checks.append({"name": "no_protocol_leak", "passed": passed, "reason": reason})

    if expect_score:
        passed, reason = check_score(response_text)
        checks.append({"name": "score", "passed": passed, "reason": reason})

    if min_executed is not None:
        passed, reason = check_min_executed(last_artifact, log_text, min_executed)
        checks.append({"name": "min_executed", "passed": passed, "reason": reason})

    metrics = compute_metrics(meta, log_text, llm_artifacts, last_artifact)
    overall = all(check["passed"] for check in checks)
    return checks, metrics, overall


def format_report(run_dir: Path, checks: list[dict], metrics: dict, overall: bool) -> str:
    lines = [f"run_dir: {run_dir}"]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{status}] {check['name']}: {check['reason']}")
    lines.append("")
    lines.append("metrics:")
    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("OVERALL: " + ("PASS" if overall else "FAIL"))
    return "\n".join(lines)


def format_json(run_dir: Path, checks: list[dict], metrics: dict, overall: bool) -> str:
    return json.dumps(
        {"run_dir": str(run_dir), "checks": checks, "metrics": metrics, "overall": overall},
        indent=2,
    )


def make_llm_artifact(
        response: str,
        prompt: str = "diff context",
        agent: dict | None = None,
        timestamp: str = "2024-01-01T00:00:00+00:00",
) -> dict:
    data = {"prompt": prompt, "response": response, "prompt_system": "system"}
    if agent is not None:
        data["agent"] = agent
    return {"id": str(uuid.uuid4()), "type": "LLM_INTERACTION", "data": data, "timestamp": timestamp}


def make_vcs_summary_artifact(text: str, timestamp: str = "2024-01-01T00:00:05+00:00") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "VCS_SUMMARY",
        "data": {"summary_comment": {"text": text}},
        "timestamp": timestamp,
    }


def write_run_dir(
        base: Path,
        meta: dict,
        log_text: str,
        llm_artifacts: list[dict],
        vcs_artifacts: list[dict],
) -> Path:
    (base / "artifacts" / "llm").mkdir(parents=True, exist_ok=True)
    (base / "artifacts" / "vcs").mkdir(parents=True, exist_ok=True)
    (base / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (base / "run.log").write_text(log_text, encoding="utf-8")
    for index, artifact in enumerate(llm_artifacts):
        (base / "artifacts" / "llm" / f"{index}.json").write_text(json.dumps(artifact), encoding="utf-8")
    for index, artifact in enumerate(vcs_artifacts):
        (base / "artifacts" / "vcs" / f"{index}.json").write_text(json.dumps(artifact), encoding="utf-8")
    return base


def expect(actual, expected, label: str) -> list[str]:
    if actual == expected:
        return []
    return [f"{label}: expected {expected!r}, got {actual!r}"]


def expect_all_passed(checks: list[dict], label: str) -> list[str]:
    failures = []
    for check in checks:
        if not check["passed"]:
            failures.append(f"{label}: expected check '{check['name']}' to pass, reason: {check['reason']}")
    return failures


def check_by_name(checks: list[dict], name: str) -> dict:
    for check in checks:
        if check["name"] == name:
            return check
    raise KeyError(name)


def run_self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:05:00+00:00",
            },
            log_text="line one\nWaiting for response\nWaiting for response\n",
            llm_artifacts=[
                make_llm_artifact(
                    "Review done.\n\nOverall score: 8/10",
                    agent={"iterations": 3, "executed_tool_calls": 3, "blocked_tool_calls": 0, "stop_reason": "final"},
                )
            ],
            vcs_artifacts=[make_vcs_summary_artifact("Review done.\n\nOverall score: 8/10")],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=True, min_executed=2, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        failures += expect(overall, True, "good scenario overall")
        failures += expect_all_passed(checks, "good scenario checks")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 1,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:01:00+00:00",
            },
            log_text="Traceback (most recent call last):\nboom\n",
            llm_artifacts=[make_llm_artifact("")],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        failures += expect(overall, False, "bad scenario overall")
        failures += expect(check_by_name(checks, "exit_code")["passed"], False, "bad scenario exit_code check")
        failures += expect(check_by_name(checks, "no_traceback")["passed"], False, "bad scenario traceback check")
        failures += expect(check_by_name(checks, "llm_response")["passed"], False, "bad scenario llm_response check")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:30+00:00",
            },
            log_text="no issues here\n",
            llm_artifacts=[make_llm_artifact('{"action": "TOOL_CALL"} leaked <TOOL_CALL>cat foo</TOOL_CALL>')],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        failures += expect(check_by_name(checks, "no_protocol_leak")["passed"], False, "leak scenario protocol check")
        failures += expect(overall, False, "leak scenario overall")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:02:00+00:00",
            },
            log_text="Agent loop summary: iterations=5 executed_tool_calls=4 blocked_tool_calls=1 stop_reason=final\n",
            llm_artifacts=[make_llm_artifact("Overall score: 7")],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=True, min_executed=3, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        failures += expect(check_by_name(checks, "min_executed")["passed"], True, "log-fallback min_executed pass")
        failures += expect(metrics["executed_tool_calls"], 4, "log-fallback metrics executed_tool_calls")

        checks_too_high, _, _ = evaluate(
            run_dir, expect_score=True, min_executed=10, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        failures += expect(check_by_name(checks_too_high, "min_executed")["passed"], False, "log-fallback min_executed fail")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:10+00:00",
            },
            log_text="nothing about agent here\n",
            llm_artifacts=[make_llm_artifact("Overall score: 7")],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=1, expect_exit=0, allow_traceback=False, expect_empty=False
        )
        stats_check = check_by_name(checks, "min_executed")
        failures += expect(stats_check["passed"], False, "unavailable min_executed fail")
        failures += expect(stats_check["reason"], "agent stats unavailable", "unavailable min_executed reason")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:10+00:00",
            },
            log_text="dry run only\n",
            llm_artifacts=[make_llm_artifact("")],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=True
        )
        failures += expect(overall, True, "expect-empty pass scenario overall")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:10+00:00",
            },
            log_text="dry run only\n",
            llm_artifacts=[make_llm_artifact("")],
            vcs_artifacts=[make_vcs_summary_artifact("unexpected summary")],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=True
        )
        failures += expect(
            check_by_name(checks, "llm_response")["passed"], False, "expect-empty fail when vcs summary present"
        )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:10+00:00",
            },
            log_text="Agent mode aborted before verifying anything, posting nothing\n",
            llm_artifacts=[],
            vcs_artifacts=[],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=True
        )
        failures += expect(overall, True, "expect-empty pass scenario with no LLM artifacts overall")
        failures += expect(
            check_by_name(checks, "llm_response")["passed"], True, "expect-empty pass check with no LLM artifacts"
        )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = write_run_dir(
            Path(tmp),
            meta={
                "docker_exit_code": 0,
                "started_at": "2024-01-01T00:00:00+00:00",
                "finished_at": "2024-01-01T00:00:10+00:00",
            },
            log_text="dry run only\n",
            llm_artifacts=[],
            vcs_artifacts=[make_vcs_summary_artifact("unexpected summary")],
        )
        checks, metrics, overall = evaluate(
            run_dir, expect_score=False, min_executed=None, expect_exit=0, allow_traceback=False, expect_empty=True
        )
        failures += expect(overall, False, "expect-empty fail with no LLM artifacts but vcs summary present overall")
        failures += expect(
            check_by_name(checks, "llm_response")["passed"],
            False,
            "expect-empty fail check with no LLM artifacts but vcs summary present",
        )

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print(f"self-test FAILED ({len(failures)} assertion(s))", file=sys.stderr)
        return 1

    print("self-test OK")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--expect-score", action="store_true")
    parser.add_argument("--min-executed", type=int, default=None)
    parser.add_argument("--expect-exit", type=int, default=0)
    parser.add_argument("--allow-traceback", action="store_true")
    parser.add_argument("--expect-empty", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.run_dir:
        parser.error("--run-dir is required unless --self-test is given")

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"run directory not found: {run_dir}", file=sys.stderr)
        return 1

    checks, metrics, overall = evaluate(
        run_dir=run_dir,
        expect_score=args.expect_score,
        min_executed=args.min_executed,
        expect_exit=args.expect_exit,
        allow_traceback=args.allow_traceback,
        expect_empty=args.expect_empty,
    )

    if args.json:
        print(format_json(run_dir, checks, metrics, overall))
    else:
        print(format_report(run_dir, checks, metrics, overall))

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
