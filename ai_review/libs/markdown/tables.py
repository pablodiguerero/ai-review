import re

SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
FENCE_RE = re.compile(r"^```")


def _split_unescaped_pipes(text: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length and text[index + 1] == "|":
            current.append("\\|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1

    cells.append("".join(current))
    return cells


def _strip_outer_pipes(text: str) -> str:
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    return text


def _row_cells(line: str) -> list[str]:
    return _split_unescaped_pipes(_strip_outer_pipes(line.strip()))


def _is_table_row_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False

    return len(_row_cells(line)) >= 2


def _is_separator_row(cells: list[str]) -> bool:
    return all(SEPARATOR_CELL_RE.fullmatch(cell.strip()) for cell in cells)


def _format_row(cells: list[str], column_count: int) -> str:
    cells = [cell.strip() for cell in cells]
    if len(cells) < column_count:
        cells = cells + [""] * (column_count - len(cells))
    elif len(cells) > column_count:
        cells = cells[:column_count]

    return "| " + " | ".join(cells) + " |"


def _normalize_table_block(block_lines: list[str]) -> list[str]:
    parsed_rows = [_row_cells(line) for line in block_lines]

    if _is_separator_row(parsed_rows[0]):
        return list(block_lines)

    column_count = len(parsed_rows[0])
    rows = [parsed_rows[0]]

    if len(parsed_rows) >= 2 and _is_separator_row(parsed_rows[1]):
        rows.append(parsed_rows[1])
        remaining = parsed_rows[2:]
    else:
        rows.append(["---"] * column_count)
        remaining = parsed_rows[1:]

    rows.extend(remaining)

    return [_format_row(cells, column_count) for cells in rows]


def normalize_markdown_tables(text: str) -> str:
    lines = text.split("\n")
    output: list[str] = []
    in_fence = False
    pending_blank_before_next = False
    index = 0
    total = len(lines)

    while index < total:
        line = lines[index]
        stripped = line.strip()

        if FENCE_RE.match(stripped):
            in_fence = not in_fence
            output.append(line)
            pending_blank_before_next = False
            index += 1
            continue

        if in_fence:
            output.append(line)
            index += 1
            continue

        if _is_table_row_candidate(line):
            start = index
            while index < total and _is_table_row_candidate(lines[index]):
                index += 1
            block_lines = lines[start:index]

            if len(block_lines) < 2 or _is_separator_row(_row_cells(block_lines[0])):
                output.extend(block_lines)
                pending_blank_before_next = False
                continue

            while output and output[-1].strip() == "":
                output.pop()
            if output:
                output.append("")

            output.extend(_normalize_table_block(block_lines))
            pending_blank_before_next = True
            continue

        if pending_blank_before_next:
            if stripped == "":
                index += 1
                continue

            output.append("")
            pending_blank_before_next = False

        output.append(line)
        index += 1

    return "\n".join(output)
