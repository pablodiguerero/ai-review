from ai_review.libs.config.review import ReviewConfig


def test_review_config_fail_on_empty_result_default() -> None:
    config = ReviewConfig()
    assert config.fail_on_empty_result is False


def test_review_config_fail_on_empty_result_can_be_enabled() -> None:
    config = ReviewConfig(fail_on_empty_result=True)
    assert config.fail_on_empty_result is True


def test_review_config_summary_header_default_is_empty() -> None:
    config = ReviewConfig()
    assert config.summary_header == ""


def test_review_config_summary_header_can_be_set() -> None:
    config = ReviewConfig(summary_header="### AI review: {model}")
    assert config.summary_header == "### AI review: {model}"


def test_review_config_summary_replace_previous_default_is_false() -> None:
    config = ReviewConfig()
    assert config.summary_replace_previous is False


def test_review_config_summary_replace_previous_can_be_enabled() -> None:
    config = ReviewConfig(summary_replace_previous=True)
    assert config.summary_replace_previous is True


def test_review_config_summary_normalize_tables_default_is_true() -> None:
    config = ReviewConfig()
    assert config.summary_normalize_tables is True


def test_review_config_summary_normalize_tables_can_be_disabled() -> None:
    config = ReviewConfig(summary_normalize_tables=False)
    assert config.summary_normalize_tables is False
