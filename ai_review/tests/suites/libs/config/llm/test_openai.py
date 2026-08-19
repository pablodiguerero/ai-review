import pytest
from pydantic import ValidationError

from ai_review.libs.config.llm.openai import RESERVED_EXTRA_BODY_KEYS, OpenAIAPI, OpenAIMetaConfig


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-5", True),
        ("gpt-5-preview", True),
        ("gpt-4.1", True),
        ("gpt-4.1-mini", True),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("gpt-3.5-turbo", False),
        ("text-davinci-003", False),
    ],
)
def test_is_v2_model_detection(model: str, expected: bool):
    meta = OpenAIMetaConfig(model=model)
    assert meta.is_v2_model is expected, f"Model {model} expected {expected} but got {meta.is_v2_model}"


def test_is_v2_model_default_false():
    meta = OpenAIMetaConfig()
    assert meta.model == "gpt-4o-mini"
    assert meta.is_v2_model is False
    assert meta.max_tokens is None


def test_openai_meta_config_api_defaults_to_auto():
    meta = OpenAIMetaConfig()
    assert meta.api is OpenAIAPI.AUTO


def test_openai_meta_config_api_parses_from_string():
    meta = OpenAIMetaConfig(api="RESPONSES")
    assert meta.api is OpenAIAPI.RESPONSES


def test_openai_meta_config_allows_stream_for_responses_api_models():
    meta = OpenAIMetaConfig(model="gpt-5", stream=True)
    assert meta.use_responses_api is True
    assert meta.stream is True


def test_openai_meta_config_allows_stream_for_chat_api_models():
    assert OpenAIMetaConfig(model="deepseek-v4-flash", stream=True).stream is True


@pytest.mark.parametrize(
    "model, api, expected",
    [
        ("gpt-5", OpenAIAPI.AUTO, True),
        ("gpt-4o-mini", OpenAIAPI.AUTO, False),
        ("gpt-5", OpenAIAPI.CHAT, False),
        ("gpt-4o-mini", OpenAIAPI.CHAT, False),
        ("gpt-5", OpenAIAPI.RESPONSES, True),
        ("deepseek-v4-flash", OpenAIAPI.RESPONSES, True),
        ("deepseek-v4-flash", OpenAIAPI.CHAT, False),
        ("deepseek-v4-flash", OpenAIAPI.AUTO, False),
    ],
)
def test_use_responses_api_matrix(model: str, api: OpenAIAPI, expected: bool):
    meta = OpenAIMetaConfig(model=model, api=api)
    assert meta.use_responses_api is expected


def test_openai_meta_config_allows_stream_for_v2_model_forced_to_chat_api():
    meta = OpenAIMetaConfig(model="gpt-5", api=OpenAIAPI.CHAT, stream=True)
    assert meta.use_responses_api is False
    assert meta.stream is True


def test_openai_meta_config_allows_stream_for_chat_model_forced_to_responses_api():
    meta = OpenAIMetaConfig(model="deepseek-v4-flash", api=OpenAIAPI.RESPONSES, stream=True)
    assert meta.use_responses_api is True
    assert meta.stream is True


def test_openai_meta_config_extra_body_defaults_to_none():
    meta = OpenAIMetaConfig()
    assert meta.extra_body is None


def test_openai_meta_config_extra_body_from_model_validate():
    meta = OpenAIMetaConfig.model_validate(
        {"model": "deepseek-v4-flash", "extra_body": {"thinking": {"type": "disabled"}}}
    )
    assert meta.extra_body == {"thinking": {"type": "disabled"}}


def test_openai_meta_config_extra_body_accepts_grok_reasoning_effort():
    meta = OpenAIMetaConfig.model_validate(
        {"model": "grok-4.5", "api": "RESPONSES", "extra_body": {"reasoning": {"effort": "low"}}}
    )
    assert meta.extra_body == {"reasoning": {"effort": "low"}}


def test_openai_meta_config_extra_body_accepts_vendor_keys():
    meta = OpenAIMetaConfig(
        model="deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}, "temperature": 0.9},
    )
    assert meta.extra_body == {"thinking": {"type": "disabled"}, "temperature": 0.9}


@pytest.mark.parametrize("key", RESERVED_EXTRA_BODY_KEYS)
def test_openai_meta_config_rejects_extra_body_reserved_keys(key: str):
    with pytest.raises(ValidationError, match=key):
        OpenAIMetaConfig(model="gpt-4o-mini", extra_body={key: "anything"})


def test_openai_meta_config_rejects_extra_body_reserved_key_among_others():
    with pytest.raises(ValidationError, match="stream_options"):
        OpenAIMetaConfig(
            model="gpt-4o-mini",
            extra_body={"thinking": {"type": "disabled"}, "stream_options": {}},
        )
