from ai_review.clients.openai.v2.schema import (
    OpenAIResponseUsageSchema,
    OpenAIInputMessageSchema,
    OpenAIResponseContentSchema,
    OpenAIResponseOutputSchema,
    OpenAIResponsesRequestSchema,
    OpenAIResponsesResponseSchema,
)


def test_first_text_returns_combined_text():
    resp = OpenAIResponsesResponseSchema(
        usage=OpenAIResponseUsageSchema(total_tokens=42, input_tokens=21, output_tokens=21),
        output=[
            OpenAIResponseOutputSchema(
                type="message",
                role="assistant",
                content=[
                    OpenAIResponseContentSchema(type="output_text", text="Hello"),
                    OpenAIResponseContentSchema(type="output_text", text=" World"),
                ],
            )
        ],
    )

    assert resp.first_text == "Hello World"


def test_first_text_empty_if_no_output():
    resp = OpenAIResponsesResponseSchema(
        usage=OpenAIResponseUsageSchema(total_tokens=0, input_tokens=0, output_tokens=0),
        output=[],
    )
    assert resp.first_text == ""


def test_first_text_ignores_non_message_blocks():
    resp = OpenAIResponsesResponseSchema(
        usage=OpenAIResponseUsageSchema(total_tokens=5, input_tokens=2, output_tokens=3),
        output=[
            OpenAIResponseOutputSchema(
                type="reasoning",
                role=None,
                content=None,
            )
        ],
    )
    assert resp.first_text == ""


def test_first_text_ignores_reasoning_block_and_returns_message_text():
    resp = OpenAIResponsesResponseSchema(
        usage=OpenAIResponseUsageSchema(total_tokens=5, input_tokens=2, output_tokens=3),
        output=[
            OpenAIResponseOutputSchema(type="reasoning", role=None, content=None),
            OpenAIResponseOutputSchema(
                type="message",
                role="assistant",
                content=[OpenAIResponseContentSchema(type="output_text", text="Hello")],
            ),
        ],
    )
    assert resp.first_text == "Hello"


def test_usage_defaults_to_zero_when_missing():
    resp = OpenAIResponsesResponseSchema.model_validate({"output": []})

    assert resp.usage.total_tokens == 0
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


def test_status_and_incomplete_details_default_to_none():
    resp = OpenAIResponsesResponseSchema.model_validate({"output": []})

    assert resp.status is None
    assert resp.incomplete_details is None


def test_status_and_incomplete_details_parse_from_payload():
    resp = OpenAIResponsesResponseSchema.model_validate(
        {"output": [], "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    )

    assert resp.status == "incomplete"
    assert resp.incomplete_details == {"reason": "max_output_tokens"}


def test_responses_request_schema_builds_ok():
    msg = OpenAIInputMessageSchema(role="user", content="hello")
    req = OpenAIResponsesRequestSchema(
        model="gpt-5",
        input=[msg],
        temperature=0.2,
        max_output_tokens=512,
        instructions="You are a helpful assistant.",
    )

    assert req.model == "gpt-5"
    assert req.input[0].role == "user"
    assert req.input[0].content == "hello"
    assert req.temperature == 0.2
    assert req.max_output_tokens == 512
    assert req.instructions == "You are a helpful assistant."


def test_responses_request_schema_allows_none_tokens():
    req = OpenAIResponsesRequestSchema(
        model="gpt-5",
        input=[OpenAIInputMessageSchema(role="user", content="test")],
    )

    dumped = req.model_dump(exclude_none=True)
    assert "max_output_tokens" not in dumped


def test_responses_request_schema_stream_defaults_to_false():
    msg = OpenAIInputMessageSchema(role="user", content="hi")
    req = OpenAIResponsesRequestSchema(model="gpt-5", input=[msg])
    assert req.stream is False


def test_responses_request_schema_stream_included_in_payload():
    msg = OpenAIInputMessageSchema(role="user", content="hi")
    req = OpenAIResponsesRequestSchema(model="gpt-5", input=[msg])
    payload = req.model_dump(exclude_none=True)
    assert payload["stream"] is False


def test_responses_request_schema_omits_text_by_default():
    msg = OpenAIInputMessageSchema(role="user", content="hi")
    req = OpenAIResponsesRequestSchema(model="gpt-5", input=[msg])

    assert req.text is None
    assert "text" not in req.model_dump(exclude_none=True)


def test_responses_request_schema_includes_text_format_when_set():
    msg = OpenAIInputMessageSchema(role="user", content="hi")
    req = OpenAIResponsesRequestSchema(
        model="gpt-5", input=[msg], text={"format": {"type": "json_object"}}
    )

    payload = req.model_dump(exclude_none=True)
    assert payload["text"] == {"format": {"type": "json_object"}}


def test_responses_request_schema_allows_and_dumps_extra_fields():
    msg = OpenAIInputMessageSchema(role="user", content="hi")
    req = OpenAIResponsesRequestSchema(
        model="grok-4.5", input=[msg], reasoning={"effort": "low"}
    )

    payload = req.model_dump(exclude_none=True)
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["model"] == "grok-4.5"
