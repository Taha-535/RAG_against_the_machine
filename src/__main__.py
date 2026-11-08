"""Command-line interface of the RAG pipeline (built with Python Fire).

Only ``index``, ``search``, ``search_dataset``, ``answer``,
``answer_dataset`` and ``evaluate`` are reachable from the command line.

``fire.Fire`` is never given a container (module, dict, class or instance).
Instead, the command name is looked up here in a private table and Fire is
handed that single function. Any other name is rejected before Fire runs.

Usage: ``uv run python -m src <command> [options]``
"""

from typing import Callable, Any
import sys

try:
    from pydantic import ValidationError
    from src.helper import format_pydantic_error
    from src.pipeline import (
        index,
        search,
        search_dataset,
        answer,
        answer_dataset,
        evaluate,
    )
    import fire
except ImportError as e:
    print(
        f"ImportError: {e}. Run `uv sync` or `make install` first!",
        file=sys.stderr,
    )
    sys.exit(1)


def _usage() -> str:
    """Build the list of available commands.

    Returns:
        A multi-line usage message.
    """
    lines = ["Usage: python -m src <command> [options]", "", "Commands:"]

    for name, func in _COMMANDS.items():
        summary = (func.__doc__ or "").strip().splitlines()[0]
        lines.append(f"  {name:<16}{summary}")

    lines += ["", "Run `python -m src <command> --help` for its options."]

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Run the command named by the first argument through Fire.

    Args:
        argv: Command-line arguments, without the program name.

    Returns:
        The process exit code.
    """
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0 if argv and argv[0] in {"-h", "--help", "help"} else 1

    command = _COMMANDS.get(argv[0])

    if command is None:
        print(f"Unknown command: {argv[0]}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 1

    try:
        fire.Fire(command, command=argv[1:], name=argv[0])
    except ValidationError as e:
        print(format_pydantic_error(e, command), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Program has been terminated!", file=sys.stderr)
        return 1
    except Exception as e:
        print(
            f"Unexpected error has occurred: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


_COMMANDS: dict[str, Callable[..., Any]] = {
    "index": index,
    "search": search,
    "search_dataset": search_dataset,
    "answer": answer,
    "answer_dataset": answer_dataset,
    "evaluate": evaluate,
}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
