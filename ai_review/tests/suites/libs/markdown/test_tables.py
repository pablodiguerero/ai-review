from ai_review.libs.markdown.tables import normalize_markdown_tables


def test_normalizes_real_review_table_without_separator_or_outer_pipes() -> None:
    raw = (
        "## Clean Code Evaluation Table\n\n"
        "Criterion | Rating | Explanation\n"
        "Naming | 4 | Mostly clear names\n"
        "Functions | 3 | Some are too long\n"
        "Comments | 5 | Well documented\n"
        "Formatting | 4 | Consistent style\n"
        "Duplication | 3 | Minor repetition\n"
        "Error Handling | 4 | Solid coverage\n"
        "Tests | 2 | Missing edge cases\n"
        "Complexity | 3 | Some deep nesting\n\n"
        "Overall this is a solid submission."
    )

    result = normalize_markdown_tables(raw)

    lines = result.splitlines()
    assert "| Criterion | Rating | Explanation |" in lines
    header_index = lines.index("| Criterion | Rating | Explanation |")
    assert lines[header_index + 1] == "| --- | --- | --- |"
    assert "| Naming | 4 | Mostly clear names |" in lines
    assert "| Complexity | 3 | Some deep nesting |" in lines
    assert "## Clean Code Evaluation Table" in result
    assert "Overall this is a solid submission." in result


def test_proper_table_is_left_byte_identical() -> None:
    text = (
        "Intro paragraph.\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n\n"
        "Outro paragraph."
    )

    assert normalize_markdown_tables(text) == text


def test_normalization_is_idempotent() -> None:
    raw = "Header | Value\nfoo | bar\nbaz | qux"

    once = normalize_markdown_tables(raw)
    twice = normalize_markdown_tables(once)

    assert once == twice


def test_text_without_tables_is_unchanged() -> None:
    text = "Just a plain review.\n\nNo tables here, only prose.\nMultiple lines too."

    assert normalize_markdown_tables(text) == text


def test_fenced_code_blocks_are_left_untouched() -> None:
    text = (
        "Some review text.\n\n"
        "```python\n"
        "a | b = 1, 2\n"
        "row | data\n"
        "```\n\n"
        "More text after."
    )

    assert normalize_markdown_tables(text) == text


def test_table_inside_fence_is_not_normalized_but_table_outside_is() -> None:
    text = (
        "A | B\n"
        "1 | 2\n\n"
        "```\n"
        "x | y\n"
        "z | w\n"
        "```"
    )

    result = normalize_markdown_tables(text)

    assert "| A | B |" in result
    assert "| --- | --- |" in result
    assert "x | y" in result
    assert "z | w" in result
    assert "| x | y |" not in result


def test_rows_with_fewer_cells_are_padded_to_header_width() -> None:
    raw = "A | B | C\n1 | 2\n3 | 4 | 5"

    result = normalize_markdown_tables(raw)
    lines = result.splitlines()

    assert lines[0] == "| A | B | C |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| 1 | 2 |  |"
    assert lines[3] == "| 3 | 4 | 5 |"


def test_rows_with_more_cells_are_truncated_to_header_width() -> None:
    raw = "A | B\n1 | 2 | 3 | 4"

    result = normalize_markdown_tables(raw)
    lines = result.splitlines()

    assert lines[0] == "| A | B |"
    assert lines[2] == "| 1 | 2 |"


def test_escaped_pipes_are_kept_and_not_treated_as_separators() -> None:
    raw = "A | B\nval\\|ue | other"

    result = normalize_markdown_tables(raw)

    assert "val\\|ue" in result
    lines = result.splitlines()
    data_row = next(line for line in lines if "val" in line)
    assert data_row == "| val\\|ue | other |"


def test_blank_lines_are_inserted_around_table_with_no_surrounding_gap() -> None:
    raw = "Intro text\nA | B\n1 | 2\nOutro text"

    result = normalize_markdown_tables(raw)

    assert result == (
        "Intro text\n\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n\n"
        "Outro text"
    )


def test_extra_blank_lines_around_table_are_collapsed_to_one() -> None:
    raw = "Intro\n\n\n\nA | B\n1 | 2\n\n\n\nOutro"

    result = normalize_markdown_tables(raw)

    assert result == "Intro\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nOutro"


def test_table_at_start_of_document_gets_no_leading_blank_line() -> None:
    raw = "A | B\n1 | 2\n\nAfter."

    result = normalize_markdown_tables(raw)

    assert result == "| A | B |\n| --- | --- |\n| 1 | 2 |\n\nAfter."
    assert not result.startswith("\n")


def test_table_at_end_of_document_gets_no_trailing_blank_line() -> None:
    raw = "Before.\n\nA | B\n1 | 2"

    result = normalize_markdown_tables(raw)

    assert result == "Before.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert not result.endswith("\n")


def test_block_left_untouched_when_first_row_looks_like_separator() -> None:
    raw = "--- | ---\nfoo | bar"

    assert normalize_markdown_tables(raw) == raw


def test_single_pipe_line_is_not_treated_as_a_table_block() -> None:
    raw = "This has a | pipe but is not a table by itself.\nNext plain line."

    assert normalize_markdown_tables(raw) == raw


def test_existing_valid_separator_row_is_preserved_and_reformatted() -> None:
    raw = "A | B\n:---: | ---:\n1 | 2"

    result = normalize_markdown_tables(raw)
    lines = result.splitlines()

    assert lines[1] == "| :---: | ---: |"
