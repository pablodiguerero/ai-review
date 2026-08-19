import pytest
from pydantic import ValidationError

from ai_review.config import settings
from ai_review.libs.config.llm.openai import OpenAIMetaConfig
from ai_review.services.llm.openai.client import OpenAILLMClient
from ai_review.services.llm.types import ChatResultSchema
from ai_review.tests.fixtures.clients.openai import FakeOpenAIV1HTTPClient, FakeOpenAIV2HTTPClient


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config")
async def test_openai_llm_chat_v1(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient
):
    result = await openai_llm_client.chat("prompt", "prompt_system")

    assert isinstance(result, ChatResultSchema)
    assert result.text == "FAKE_OPENAI_V1_RESPONSE"
    assert result.total_tokens == 12
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 7

    assert fake_openai_v1_http_client.calls[0][0] == "chat"


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_chat_v2(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient
):
    result = await openai_llm_client.chat("prompt", "prompt_system")

    assert isinstance(result, ChatResultSchema)
    assert result.text == "FAKE_OPENAI_V2_RESPONSE"
    assert result.total_tokens == 20
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 10

    assert fake_openai_v2_http_client.calls[0][0] == "chat"


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config")
async def test_openai_llm_client_chat_v1_sends_stream_options_when_streaming(
        monkeypatch: pytest.MonkeyPatch,
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
):
    monkeypatch.setattr(settings.llm.meta, "stream", True)

    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v1_http_client.calls[0][1]["request"]
    assert request.stream is True
    assert request.stream_options.include_usage is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config")
async def test_openai_llm_client_chat_v1_omits_stream_options_by_default(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
):
    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v1_http_client.calls[0][1]["request"]
    assert request.stream is False
    assert request.stream_options is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config_forced_chat")
async def test_openai_llm_client_api_chat_forces_v1_for_a_v2_model(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    result = await openai_llm_client.chat("prompt", "prompt_system")

    assert result.text == "FAKE_OPENAI_V1_RESPONSE"
    assert fake_openai_v1_http_client.calls[0][0] == "chat"
    assert fake_openai_v2_http_client.calls == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config_forced_responses")
async def test_openai_llm_client_api_responses_forces_v2_for_a_chat_model(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    result = await openai_llm_client.chat("prompt", "prompt_system")

    assert result.text == "FAKE_OPENAI_V2_RESPONSE"
    assert fake_openai_v2_http_client.calls[0][0] == "chat"
    assert fake_openai_v1_http_client.calls == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_sends_json_format_when_json_mode(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    await openai_llm_client.chat("prompt", "prompt_system", json_mode=True)

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    assert request.text == {"format": {"type": "json_object"}}


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_omits_text_format_by_default(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    await openai_llm_client.chat("prompt", "prompt_system")

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    assert request.text is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config")
async def test_openai_llm_client_chat_v1_merges_extra_body_into_payload(
        monkeypatch: pytest.MonkeyPatch,
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
):
    monkeypatch.setattr(settings.llm.meta, "extra_body", {"thinking": {"type": "disabled"}})

    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v1_http_client.calls[0][1]["request"]
    payload = request.model_dump(exclude_none=True)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "gpt-4o-mini"
    assert payload["max_tokens"] == 1200
    assert len(payload["messages"]) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v1_http_client_config")
async def test_openai_llm_client_chat_v1_payload_unchanged_when_extra_body_is_none(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v1_http_client: FakeOpenAIV1HTTPClient,
):
    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v1_http_client.calls[0][1]["request"]
    payload = request.model_dump(exclude_none=True)
    assert set(payload.keys()) == {"model", "messages", "max_tokens", "temperature", "stream"}


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_merges_extra_body_into_payload(
        monkeypatch: pytest.MonkeyPatch,
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    monkeypatch.setattr(settings.llm.meta, "extra_body", {"reasoning": {"effort": "low"}})

    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    payload = request.model_dump(exclude_none=True)
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["model"] == "gpt-5"
    assert len(payload["input"]) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_payload_unchanged_when_extra_body_is_none(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    payload = request.model_dump(exclude_none=True)
    assert set(payload.keys()) == {"model", "input", "stream", "temperature", "max_output_tokens"}


def test_openai_meta_config_rejects_extra_body_overriding_model():
    with pytest.raises(ValidationError, match="model"):
        OpenAIMetaConfig(model="gpt-4o-mini", extra_body={"model": "overridden-model"})


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_sends_stream_flag_when_streaming(
        monkeypatch: pytest.MonkeyPatch,
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    monkeypatch.setattr(settings.llm.meta, "stream", True)

    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    assert request.stream is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_v2_http_client_config")
async def test_openai_llm_client_chat_v2_omits_stream_by_default(
        openai_llm_client: OpenAILLMClient,
        fake_openai_v2_http_client: FakeOpenAIV2HTTPClient,
):
    await openai_llm_client.chat(prompt="prompt", prompt_system="system")

    request = fake_openai_v2_http_client.calls[0][1]["request"]
    assert request.stream is False
