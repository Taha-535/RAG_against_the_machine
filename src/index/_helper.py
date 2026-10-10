"""Helpers shared by the indexing steps."""

from typing import Any
import sys


def get_chunk_content(
    curr_file: tuple[str, str], chunk: dict[str, Any]
) -> tuple[tuple[str, str], str]:
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
        try:
            with open(chunk["file_path"]) as f:
                curr_file = (chunk["file_path"], f.read())
        except IOError as e:
            print(
                f"{type(e)}: {e}: "
                f"Couldn't read file {chunk['file_path']}"
            )
            sys.exit(1)

    return curr_file, curr_file[1][
        chunk["first_character_index"]:chunk["last_character_index"] + 1
    ]
