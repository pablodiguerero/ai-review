import pytest

from ai_review.config import settings
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
