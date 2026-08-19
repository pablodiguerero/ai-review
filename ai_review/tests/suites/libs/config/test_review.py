from ai_review.libs.config.review import ReviewConfig


def test_review_config_fail_on_empty_result_default() -> None:
    config = ReviewConfig()
    assert config.fail_on_empty_result is False


def test_review_config_fail_on_empty_result_can_be_enabled() -> None:
    config = ReviewConfig(fail_on_empty_result=True)
    assert config.fail_on_empty_result is True
