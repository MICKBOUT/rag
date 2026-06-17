from pathlib import Path

import re
from .parser_common import IndexChunk, display_path


def get_ready_to_index_md_file(
        file_name: str | Path,
        inherited_headings: list[str] | None = None,
) -> list[IndexChunk]:
    """
    Read a Markdown file and produce ready-to-index chunks for its content.
    Handles simple recursion protections and returns sliding-window chunks
    for the file range.

    Args:
        file_name (str | Path): Path to the markdown file.
        inherited_headings (list[str] | None): Optional heading path
            inherited from callers.

    Returns:
        list[IndexChunk]: Chunks produced for the Markdown file, or an
            empty list on error or if the file is empty.
    """

    def _line_starts(source: str) -> list[int]:
        """
        Compute a list of character indices where each line in source begins.

        Args:
            source (str): File source text.

        Returns:
            list[int]: 1-to-1 list of start offsets for each line; first entry
                is 0.
        """
        starts = [0]
        for match in re.finditer(r"\n", source):
            starts.append(match.end())
        return starts

    def _anchor_bounds(
            source: str, anchor: str | None) -> tuple[int, int]:
        """
        Determine start and end line bounds for a source slice optionally
        delimited by markers.
        If anchor is None, returns bounds that cover the entire document.
        If anchors are present
        in the document, this function scans marker lines to find matching
        start/end markers.

        Args:
            source (str): Full file source text.
            anchor (str | None): Optional anchor name to limit the slice.

        Returns:
            tuple[int, int]: (start_line, end_line) inclusive 1-based line
                numbers describing the slice.
        """
        marker_re = re.compile(
           r'^\s*(?:)?\s*$'
        )

        if anchor is None:
            return 1, len(source.splitlines(keepends=True))

        lines = source.splitlines(keepends=True)
        start_line = 1
        end_line = len(lines)
        found_start = False
        for idx, line in enumerate(lines, start=1):
            match = marker_re.match(line)
            if match is None:
                continue
            marker_kind = match.group("kind")
            marker_name = match.group("name")
            if marker_kind == "start" and marker_name == anchor:
                start_line = idx + 1
                found_start = True
            elif (
                marker_kind == "end"
                and marker_name == anchor
                and found_start
            ):
                end_line = idx - 1
                break
        return start_line, end_line

    def _heading_at_offset(
            lines: list[str],
            line_starts: list[int],
            char_offset: int) -> list[str]:
        """
        Compute a list of Markdown headings active at a given character offset.
        Scans lines in order and returns the hierarchical heading titles that
        precede the offset.

        Args:
            lines (list[str]): File lines (with or without line endings).
            line_starts (list[int]): Start character offsets for each line.
            char_offset (int): Absolute character offset for which to compute
                the heading stack.

        Returns:
            list[str]: Ordered list of heading titles representing the active
                heading path at offset.
        """
        heading_re = re.compile(r'^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$')
        active: list[str] = []

        for line_no, line in enumerate(lines, start=1):
            if line_starts[line_no - 1] >= char_offset:
                break
            m = heading_re.match(line)
            if m:
                level = len(m.group("hashes"))
                title = m.group("title").strip()
                active = active[:level - 1] + [title]
        return active

    def _sliding_window_md_chunks(
            file_path: Path,
            source: str,
            line_starts: list[int],
            lines: list[str],
            start_line: int,
            end_line: int,
            inherited_headings: list[str],
    ) -> list[IndexChunk]:
        """
        Produce sliding-window IndexChunks for a range of Markdown source.
        Slices the given source range into overlapping windows suitable for
        retrieval/indexing, attaches heading context, and computes absolute
        character offsets for each chunk.

        Args:
            file_path (Path): Path to the markdown file.
            source (str): Full file source text.
            line_starts (list[int]): Start character offsets for each line.
            lines (list[str]): File lines (with line endings preserved).
            start_line (int): 1-based starting line for the range to window.
            end_line (int): 1-based ending line for the range to window.
            inherited_headings (list[str]): Headings inherited from parent
                context.

        Returns:
            list[IndexChunk]: List of markdown IndexChunk objects covering the
                requested range.
        """
        md_window = 1000
        md_stride = 500

        if not lines:
            return []

        file_str = display_path(file_path)
        range_start = line_starts[start_line - 1]
        last_line_text = lines[end_line - 1].rstrip("\r\n")
        range_end = line_starts[end_line - 1] + len(last_line_text)

        source_slice = source[range_start:range_end]
        if not source_slice.strip():
            return []

        chunks: list[IndexChunk] = []
        window = md_window
        stride = md_stride
        pos = 0
        total = len(source_slice)

        while pos < total:
            win_end = min(pos + window, total)

            newline_pos = source_slice.find("\n", win_end)
            if newline_pos != -1 and newline_pos < win_end + 120:
                win_end = newline_pos + 1

            chunk_text = source_slice[pos:win_end]
            if not chunk_text.strip():
                pos += stride
                continue

            abs_start = range_start + pos
            abs_end = range_start + win_end - 1

            heading_path = (
                inherited_headings +
                _heading_at_offset(lines, line_starts, abs_start)
            )
            sep = " > "
            if heading_path:
                bm25_text = (
                    "FILE: " + file_str + "\n"
                    "HEADING: " + sep.join(heading_path) + "\n\n"
                    + chunk_text.strip()
                )
            else:
                bm25_text = "FILE: " + file_str + "\n\n" + chunk_text.strip()

            chunks.append(IndexChunk(
                text=bm25_text,
                file_path=file_str,
                first_character_index=abs_start,
                last_character_index=abs_end,
                kind='markdown',
                heading_path=heading_path,
            ))

            if win_end >= total:
                break
            pos += stride

        return chunks

    file_path = Path(file_name).resolve()
    if inherited_headings is None:
        inherited_headings = []
    active_stack = set()

    visit_key = (file_path, None)
    if visit_key in active_stack:
        return []
    active_stack.add(visit_key)
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        active_stack.remove(visit_key)
        return []

    try:
        if not source:
            return []

        line_starts = _line_starts(source)
        start_line, end_line = _anchor_bounds(source, None)
        lines = source.splitlines(keepends=True)
        if not lines:
            return []

        return _sliding_window_md_chunks(
            file_path,
            source,
            line_starts,
            lines,
            start_line,
            end_line,
            inherited_headings,
        )
    finally:
        active_stack.remove(visit_key)
