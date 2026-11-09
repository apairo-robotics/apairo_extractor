"""``apairo ros``: the commands of apairo_ros under the package's name.

    apairo ros extract                  # interactive TUI
    apairo ros extract -i bags/ -l      # the bags and their topics
    apairo ros extract -i bags/ -o dataset/ -t /lidar /imu /tf

``apairo ros extract`` is ``apairo extractor`` (also installed as
``apairo-extractor``): both names stay. The command is installed as
``apairo-ros`` too, and is found by apairo's CLI as ``apairo ros``.
"""

from __future__ import annotations

import sys

__all__ = ["main"]

_COMMANDS = {
    "extract": "ROS bags to an apairo dataset (with no argument, an interactive TUI)",
}


def _usage(prog: str) -> str:
    width = max(len(name) for name in _COMMANDS)
    lines = [
        f"usage: {prog} <command> [options]",
        "",
        "ROS for apairo.",
        "",
        "commands:",
        *(f"  {name:<{width}}  {text}" for name, text in _COMMANDS.items()),
        "",
        f"Run '{prog} <command> --help' for a command's options.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None, prog: str = "apairo ros") -> int:
    """Dispatch ``apairo ros <command> ...`` to the command's own parser."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(prog))
        return 0
    if argv[0] == "--version":
        from apairo_ros import __version__

        print(f"apairo-ros {__version__}")
        return 0
    command, rest = argv[0], argv[1:]
    if command == "extract":
        from apairo_ros.cli import main as extract

        extract(rest, prog=f"{prog} extract")
        return 0
    print(f"{prog}: unknown command {command!r}\n\n{_usage(prog)}", file=sys.stderr)
    return 2


def console() -> None:
    """The ``apairo-ros`` console script."""
    sys.exit(main(prog="apairo-ros"))
