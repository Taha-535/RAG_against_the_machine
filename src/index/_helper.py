from typing import Optional


def retrieve_chunk_content(
    curr_file: tuple[str, Optional[str]], chunk: dict[str, any]
):

    if curr_file[0] != chunk["file_path"]:
        if curr_file[1] is None:
            return

        try:
            with open(chunk["file_path"]) as f:
                curr_file = (chunk["file_path"], f.read())
        except IOError as e:
            print(
                (
                    f"[WARNING] {type(e)}: {e}: "
                    f"Couldn't read file {chunk['file_path']}",
                )
            )
            curr_file = (chunk["file_path"], None)
            return

    return curr_file[1][
        chunk["first_character_index"]:chunk["last_character_index"] + 1
    ]
