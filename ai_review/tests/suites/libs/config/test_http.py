from pydantic import HttpUrl, SecretStr

from ai_review.libs.config.http import HTTPClientConfig, HTTPClientWithTokenConfig


def test_http_client_config_defaults():
    config = HTTPClientConfig(api_url=HttpUrl("https://api.openai.com/v1"))

    assert config.verify is True
    assert config.timeout == 120
    assert config.connect_timeout == 10
    assert config.proxy_url is None
    assert config.api_url_value == "https://api.openai.com/v1"
    assert config.proxy_url_value is None


def test_http_client_config_overrides_connect_timeout():
    config = HTTPClientConfig(api_url=HttpUrl("https://api.openai.com/v1"), connect_timeout=2.5)

    assert config.connect_timeout == 2.5
    assert config.timeout == 120


def test_http_client_config_proxy_url_value():
    config = HTTPClientConfig(
        api_url=HttpUrl("https://api.openai.com/v1"),
        proxy_url=HttpUrl("https://proxy.example.com"),
    )

    assert config.proxy_url_value == "https://proxy.example.com/"


def test_http_client_with_token_config_exposes_secret_value():
    config = HTTPClientWithTokenConfig(
        api_url=HttpUrl("https://api.openai.com/v1"),
        api_token=SecretStr("fake-token"),
    )

    assert config.api_token_value == "fake-token"
    assert config.connect_timeout == 10
