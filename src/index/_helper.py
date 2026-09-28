"""Helpers shared by the indexing steps."""

from typing import Any, Optional


def retrieve_chunk_content(
    curr_file: tuple[str, Optional[str]], chunk: dict[str, Any]
) -> Optional[str]:
    """Read the text covered by a chunk.

    The file of the chunk is read from disk when it differs from the
    ``curr_file`` (path, content) pair.

    Args:
        curr_file: A ``(path, content)`` pair holding the last file read.
        chunk: The chunk, as a dictionary with ``file_path``,
            ``first_character_index`` and ``last_character_index`` keys.

    Returns:
        The text of the chunk, or ``None`` if the file cannot be read.
    """
    if curr_file[0] != chunk["file_path"]:
        if curr_file[1] is None:
            return None

        try:
            with open(chunk["file_path"]) as f:
                curr_file = (chunk["file_path"], f.read())
        except IOError as e:
            print(
                f"[WARNING] {type(e)}: {e}: "
                f"Couldn't read file {chunk['file_path']}"
            )
            return None

    content = curr_file[1]
    if content is None:
        return None

    return content[
        chunk["first_character_index"]:chunk["last_character_index"] + 1
    ]
