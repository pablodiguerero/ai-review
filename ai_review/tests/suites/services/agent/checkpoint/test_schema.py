from ai_review.services.agent.checkpoint.schema import AgentCheckpointSchema, AgentCheckpointStage
from ai_review.services.agent.loop.schema import AgentAction, AgentStepSchema, AgentTraceSchema


def test_checkpoint_round_trip_survives_json_dump_and_reload_with_real_loop_traces() -> None:
    tool_call_trace = AgentTraceSchema(
        step=AgentStepSchema(action=AgentAction.TOOL_CALL, command="rg foo src"),
        iteration=1,
        raw_output='{"action":"TOOL_CALL","command":"rg foo src"}',
        tool_output="src/foo.py:1: foo",
        total_tokens=30,
        prompt_tokens=10,
        completion_tokens=20,
    )
    duplicate_blocked_trace = AgentTraceSchema(
        step=AgentStepSchema(action=AgentAction.TOOL_CALL, command="rg foo src"),
        warning="Duplicate tool call blocked: rg foo src",
        iteration=2,
        raw_output='{"action":"TOOL_CALL","command":"rg foo src"}',
        total_tokens=15,
        prompt_tokens=5,
        completion_tokens=10,
    )
    final_trace = AgentTraceSchema(
        step=AgentStepSchema(action=AgentAction.FINAL, content="Review complete, no issues found."),
        iteration=3,
        raw_output='{"action":"FINAL","content":"Review complete, no issues found."}',
        total_tokens=12,
        prompt_tokens=4,
        completion_tokens=8,
    )

    checkpoint = AgentCheckpointSchema(
        key="1:2:gpt-4o-mini:#ai-review-summary",
        head_sha="deadbeef",
        round=2,
        stage=AgentCheckpointStage.REVIEWED,
        traces=[tool_call_trace, duplicate_blocked_trace, final_trace],
        executed_tool_calls=1,
        blocked_tool_calls=1,
        iterations=3,
        context_used=18,
        signatures=["rg foo src"],
        prior_synopsis="- round evidence: rg bar src -> no matches",
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-20T00:00:00+00:00",
    )

    dumped = checkpoint.model_dump_json()
    reloaded = AgentCheckpointSchema.model_validate_json(dumped)

    assert reloaded == checkpoint


def test_checkpoint_stage_serializes_as_plain_string_value() -> None:
    checkpoint = AgentCheckpointSchema(
        key="k1",
        stage=AgentCheckpointStage.NEEDS_FINAL,
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    assert checkpoint.model_dump()["stage"] == "needs_final"
    assert '"stage":"needs_final"' in checkpoint.model_dump_json()


def test_checkpoint_defaults_are_sensible_for_a_fresh_round() -> None:
    checkpoint = AgentCheckpointSchema(
        key="k1",
        stage=AgentCheckpointStage.INVESTIGATING,
        created_at="2026-08-19T00:00:00+00:00",
        updated_at="2026-08-19T00:00:00+00:00",
    )

    assert checkpoint.head_sha == ""
    assert checkpoint.round == 0
    assert checkpoint.traces == []
    assert checkpoint.signatures == []
    assert checkpoint.executed_tool_calls == 0
    assert checkpoint.blocked_tool_calls == 0
    assert checkpoint.iterations == 0
    assert checkpoint.context_used == 0
    assert checkpoint.prior_synopsis == ""


def test_tool_call_step_round_trip_preserves_omitted_content() -> None:
    step = AgentStepSchema(action=AgentAction.TOOL_CALL, command="ls")

    reloaded = AgentStepSchema.model_validate_json(step.model_dump_json())

    assert reloaded == step
    assert reloaded.content == ""


def test_final_step_round_trip_preserves_omitted_command() -> None:
    step = AgentStepSchema(action=AgentAction.FINAL, content="done")

    reloaded = AgentStepSchema.model_validate_json(step.model_dump_json())

    assert reloaded == step
    assert reloaded.command == ""
